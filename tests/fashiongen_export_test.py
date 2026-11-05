''' Checks fashiongen_export.py on small HDF5 files built like those of the FashionGen release: products with one
to six views and gaps, a product stored twice in a row (the same photos), a Latin-1 brand, a description with a
comma and quotes; and that it stops for rows of a product that disagree, a repeated view with another photo and a
product in both files. Needs h5py (as fashiongen_export.py); skipped without it.

    python tests/fashiongen_export_test.py
'''

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIELDS = ('name', 'description', 'gender', 'department', 'category', 'subcategory', 'season', 'brand',
          'composition', 'concat_description')


def write_release(path, products, image_seed):
    ''' One HDF5 file of the release: products is [(id, fields, poses)], each product one row per pose. '''
    import h5py
    import numpy as np
    rng = np.random.RandomState(image_seed)
    rows = [(pid, fields, pose) for pid, fields, poses in products for pose in poses]
    photos = {}
    with h5py.File(path, 'w') as f:
        n = len(rows)
        f['index'] = np.arange(n).reshape(n, 1)
        f['index_2'] = np.arange(n)
        f['input_productID'] = np.array([r[0] for r in rows], dtype=np.int32).reshape(n, 1)
        f['input_pose'] = np.array([r[2].encode() for r in rows], dtype='S40').reshape(n, 1)
        for k in FIELDS:
            f['input_' + k] = np.array([r[1][k] for r in rows], dtype='S800').reshape(n, 1)
        f['input_msrpUSD'] = np.array([r[1]['msrpUSD'] for r in rows], dtype=np.float32).reshape(n, 1)
        images = np.zeros((n, 16, 16, 3), dtype=np.uint8)
        for i, (pid, _, pose) in enumerate(rows):
            key = (pid, pose)
            if key not in photos:
                photos[key] = rng.randint(0, 256, (16, 16, 3)).astype(np.uint8)
            images[i] = photos[key]
        f['input_image'] = images
    return photos


def fields(name, brand=b'Acne Studios', description=b'Long sleeve shirt in black. Spread collar.'):
    return {'name': name, 'description': description, 'gender': b'Men', 'department': b'CLOTHING',
            'category': b'SHIRTS', 'subcategory': b'SHIRTS', 'season': b'SS2017', 'brand': brand,
            'composition': b'100% cotton.', 'concat_description': description + b' Slim-fit trousers in navy.',
            'msrpUSD': 99.99}


def run(h5_dir, out):
    return subprocess.run([sys.executable, os.path.join(REPO, 'fashiongen_export.py'), '-h5_dir', h5_dir, '-output',
                           out, '-workers', '2'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True)


def main():
    try:
        import h5py  # noqa: F401
        import numpy as np
        from PIL import Image
    except ImportError:
        print('skipped: h5py is not installed')
        return
    root = tempfile.mkdtemp()
    failures = []
    try:
        h5_dir = os.path.join(root, 'release')
        os.makedirs(h5_dir)
        poses = lambda *k: ['id_gridfs_%d' % i for i in k]
        white = fields(b'White Shirt', description=b'Shirt in "white", with a comma.')
        train = [(1001, fields(b'Black Shirt'), poses(1, 2, 3, 4)),
                 (1002, fields(b'Navy Shirt', brand=u'Chlo\xe9'.encode('latin-1')), poses(2, 1, 4)),  # a gap, unsorted
                 (1003, white, poses(1, 2, 3, 4, 5, 6)),
                 (1003, white, poses(1, 2, 3, 4, 5, 6)),  # stored twice
                 (1004, fields(b'Grey Shirt'), poses(3))]
        validation = [(2001, fields(b'Red Shirt'), poses(1, 2, 3, 4, 5))]
        photos = write_release(os.path.join(h5_dir, 'fashiongen_256_256_train.h5'), train, 0)
        photos.update(write_release(os.path.join(h5_dir, 'fashiongen_256_256_validation.h5'), validation, 1))
        out = os.path.join(root, 'export')
        result = run(h5_dir, out)
        if result.returncode != 0:
            failures.append('the export failed:\n' + result.stdout)
        else:
            with io.open(os.path.join(out, 'products.csv'), encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            expected = [('1001', 'train', 'id_gridfs_1 id_gridfs_2 id_gridfs_3 id_gridfs_4'),
                        ('1002', 'train', 'id_gridfs_1 id_gridfs_2 id_gridfs_4'),
                        ('1003', 'train', ' '.join(poses(1, 2, 3, 4, 5, 6))), ('1004', 'train', 'id_gridfs_3'),
                        ('2001', 'validation', ' '.join(poses(1, 2, 3, 4, 5)))]
            if [(r['product_id'], r['split'], r['poses']) for r in rows] != expected:
                failures.append('products.csv rows: %s' % [(r['product_id'], r['split'], r['poses']) for r in rows])
            if rows[1]['brand'] != u'Chlo\xe9':
                failures.append('Latin-1 brand read as %r' % rows[1]['brand'])
            if rows[2]['description'] != 'Shirt in "white", with a comma.' or rows[0]['msrpUSD'] != '99.99':
                failures.append('fields: %r, %r' % (rows[2]['description'], rows[0]['msrpUSD']))
            n_images = 0
            for (pid, pose), photo in photos.items():
                path = os.path.join(out, 'images', str(pid), pose + '.png')
                n_images += 1
                if not os.path.exists(path) or not np.array_equal(np.asarray(Image.open(path)), photo):
                    failures.append('image %s' % path)
            listed = sum(len(files) for _, _, files in os.walk(os.path.join(out, 'images')))
            record = json.load(io.open(os.path.join(out, 'export.json'), encoding='utf-8'))
            if listed != n_images or record['images'] != n_images or record['products'] != 5 \
                    or record['splits']['train']['repeated_rows_skipped'] != 6:
                failures.append('counts: %d files, %d photos, export.json %s' % (listed, n_images, record))
        # the export stops for inconsistent data
        cases = {'rows of a product that disagree': [(1001, fields(b'Black Shirt'), poses(1)),
                                                     (1001, fields(b'Black'), poses(2))],
                 'a repeated view with another photo': None,
                 'a product in both files': 'both'}
        for case, products in cases.items():
            bad = os.path.join(root, 'bad_' + case.replace(' ', '_'))
            os.makedirs(bad)
            if case == 'a repeated view with another photo':
                import h5py
                write_release(os.path.join(bad, 'fashiongen_256_256_train.h5'), [(1001, fields(b'A'), poses(1, 1))], 0)
                with h5py.File(os.path.join(bad, 'fashiongen_256_256_train.h5'), 'a') as f:
                    f['input_image'][1] = 255 - f['input_image'][0]
                write_release(os.path.join(bad, 'fashiongen_256_256_validation.h5'), validation, 1)
            elif products == 'both':
                write_release(os.path.join(bad, 'fashiongen_256_256_train.h5'), validation, 0)
                write_release(os.path.join(bad, 'fashiongen_256_256_validation.h5'), validation, 1)
            else:
                write_release(os.path.join(bad, 'fashiongen_256_256_train.h5'), products, 0)
                write_release(os.path.join(bad, 'fashiongen_256_256_validation.h5'), validation, 1)
            result = run(bad, os.path.join(bad, 'export'))
            if result.returncode == 0:
                failures.append('no stop for ' + case)
    finally:
        shutil.rmtree(root)
    print('\n'.join(failures) if failures else 'all passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
