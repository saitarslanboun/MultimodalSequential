''' Converts the FashionGen challenge release (fashiongen_256_256_train.h5 and fashiongen_256_256_validation.h5)
into the files that fashiongen_data.py reads: one row per product, and one image file per view.

    python fashiongen_export.py -h5_dir DIR -output OUT [-workers N]

It needs h5py (pip install h5py), which the experiments themselves do not. The HDF5 files hold one row per
photo: a product has one to six photos, one per view (input_pose id_gridfs_1 to id_gridfs_6), and each of its
rows repeats its text and attributes. Some products are stored twice in a row, with the same photos. The export
writes
  OUT/products.csv     one row per product, in the order of the release (its training file first):
                       product_id, split (the file of the release: train or validation), the text and attribute
                       fields of the rows without their input_ prefix, and poses: the views of the product, in
                       pose order
  OUT/images/<product_id>/<pose>.png   the 256x256 photo of each view, saved without loss (each file is read
                       back and compared with the photo of the release)
  OUT/export.json      the counts, the source files and the versions
It stops if the rows of a product disagree on a field, or if a repeated view is not the same photo. The text is
read as UTF-8, or as Latin-1 where it is not UTF-8 (e.g. the brand "Chloé"), and written unchanged otherwise.
'''

import argparse
import collections
import csv
import io
import json
import multiprocessing
import os
import sys

FILES = (('train', 'fashiongen_256_256_train.h5'), ('validation', 'fashiongen_256_256_validation.h5'))
# The fields of the release that products.csv keeps, without the prefix input_ (input_image is the photo,
# input_pose the view, index and index_2 the row numbers).
FIELDS = ('name', 'description', 'gender', 'department', 'category', 'subcategory', 'season', 'brand', 'msrpUSD',
          'composition', 'concat_description')
HEADER = ('product_id', 'split') + FIELDS + ('poses',)
CHUNK = 500  # photos per task of a worker process


def decode(value):
    ''' A string of the release: UTF-8, or Latin-1 where it is not valid UTF-8. '''
    try:
        return value.decode('utf-8')
    except UnicodeDecodeError:
        return value.decode('latin-1')


def pose_number(pose):
    ''' 3 for id_gridfs_3. '''
    return int(pose.rsplit('_', 1)[1])


def read_rows(path):
    ''' The product id, pose and fields (strings; the price as NumPy prints it) of every row of a file. '''
    import h5py
    with h5py.File(path, 'r') as f:
        ids = [int(x) for x in f['input_productID'][()].reshape(-1)]
        poses = [decode(x) for x in f['input_pose'][()].reshape(-1)]
        fields = {}
        for name in FIELDS:
            values = f['input_' + name][()].reshape(-1)
            fields[name] = [str(x) if name == 'msrpUSD' else decode(x) for x in values]
    return ids, poses, fields


_open_files = {}


def _export(task):
    ''' Writes the photos of a task (the file of the release, [(row, path)]) and reads them back. '''
    import h5py
    import numpy as np
    from PIL import Image
    path, items = task
    if path not in _open_files:
        _open_files[path] = h5py.File(path, 'r')
    images = _open_files[path]['input_image']
    for row, out in items:
        photo = images[row]
        Image.fromarray(photo).save(out, format='PNG')
        with Image.open(out) as saved:
            if saved.mode != 'RGB' or not np.array_equal(np.asarray(saved), photo):
                raise RuntimeError('{} differs from row {} of {}'.format(out, row, path))
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-h5_dir', required=True, help='the directory of the two HDF5 files of the release')
    parser.add_argument('-output', required=True, help='the directory to write (products.csv, images/, export.json)')
    parser.add_argument('-workers', type=int, default=8, help='processes that write the images')
    opt = parser.parse_args()
    if os.path.exists(os.path.join(opt.output, 'products.csv')):
        parser.error('{} already holds an export'.format(opt.output))
    import h5py
    import numpy as np
    import PIL

    products = collections.OrderedDict()  # product id -> split, fields, {pose: (file, row)}
    tasks, counts = [], collections.OrderedDict()
    for split, name in FILES:
        path = os.path.join(opt.h5_dir, name)
        ids, poses, fields = read_rows(path)
        rows = collections.OrderedDict()
        for row, product in enumerate(ids):
            rows.setdefault(product, []).append(row)
        duplicates = 0
        with h5py.File(path, 'r') as f:
            images = f['input_image']
            for product, product_rows in rows.items():
                if product in products:
                    sys.exit('product {} is in both files of the release'.format(product))
                values = [tuple(fields[k][r] for k in FIELDS) for r in product_rows]
                if len(set(values)) > 1:
                    sys.exit('the rows {} of product {} in {} have different fields'.format(
                        product_rows, product, name))
                views = collections.OrderedDict()
                for r in product_rows:
                    if poses[r] in views:  # a repeated view: the same photo as the first one
                        if not np.array_equal(images[r], images[views[poses[r]]]):
                            sys.exit('rows {} and {} of {} are the view {} of product {}, with different photos'.format(
                                views[poses[r]], r, name, poses[r], product))
                        duplicates += 1
                    else:
                        views[poses[r]] = r
                order = sorted(views, key=pose_number)
                products[product] = (split, values[0], order)
                directory = os.path.join(opt.output, 'images', str(product))
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                tasks += [(path, views[p], os.path.join(directory, p + '.png')) for p in order]
        counts[split] = collections.OrderedDict([
            ('rows', len(ids)), ('products', len(rows)), ('views', len(ids) - duplicates),
            ('repeated_rows_skipped', duplicates)])
        print('[Info] {}: {} rows, {} products, {} views; {} repeated rows (the same photos) skipped'.format(
            name, len(ids), len(rows), len(ids) - duplicates, duplicates))

    with io.open(os.path.join(opt.output, 'products.csv'), 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(HEADER)
        for product, (split, values, order) in products.items():
            writer.writerow((str(product), split) + values + (' '.join(order),))

    chunks = []
    for path in sorted(set(t[0] for t in tasks)):
        items = sorted((row, out) for p, row, out in tasks if p == path)
        chunks += [(path, items[i:i + CHUNK]) for i in range(0, len(items), CHUNK)]
    written = 0
    pool = multiprocessing.Pool(opt.workers)
    try:
        for n in pool.imap_unordered(_export, chunks):
            written += n
    finally:
        pool.close()
        pool.join()
    print('[Info] {} products, {} images written and read back to {}'.format(len(products), written, opt.output))

    record = collections.OrderedDict([
        ('products', len(products)), ('images', written), ('splits', counts),
        ('source', [collections.OrderedDict([('file', name),
                                             ('bytes', os.path.getsize(os.path.join(opt.h5_dir, name)))])
                    for _, name in FILES]),
        ('versions', collections.OrderedDict([('python', sys.version.split()[0]), ('h5py', h5py.__version__),
                                              ('numpy', np.__version__), ('pillow', PIL.__version__)]))])
    with io.open(os.path.join(opt.output, 'export.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(record, indent=2) + u'\n')


if __name__ == '__main__':
    main()
