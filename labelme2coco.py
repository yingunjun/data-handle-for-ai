#!/usr/bin/env python

import argparse
import collections
import datetime
import json
import os.path as osp
import sys
import os
import glob
import random
import shutil

import numpy as np
import PIL.Image
import labelme


work_dir = '/media/job/myjob/new表计/new_orign/fenhebiao/segment'

if False:

	tmp = os.path.join(work_dir,'jsons')

	json_dir = tmp
	train_dir = os.path.join(tmp,'train')
	valid_dir = os.path.join(tmp,'valid')

	if not os.path.exists(train_dir):
		os.makedirs(train_dir)
	else:
		shutil.rmtree(train_dir)
		os.makedirs(train_dir)

	if not os.path.exists(valid_dir):
		os.makedirs(valid_dir)
	else:
		shutil.rmtree(valid_dir)
		os.makedirs(valid_dir)

	file_list = glob.glob(os.path.join(json_dir,'*.png')) + glob.glob(os.path.join(json_dir,'*.jpg')) + glob.glob(os.path.join(json_dir,'*.jpeg'))
	random.shuffle(file_list)

	num = len(file_list) // 10
	# num = 0

	for i, file in enumerate(file_list):
		json_file = file.replace('.png','.json').replace('.jpg','.json').replace('.jpeg','.json')
		if i < num:
			dst_dir = valid_dir
		else:
			dst_dir = train_dir

		shutil.copy(file,dst_dir)
		shutil.copy(json_file,dst_dir)

else:

	input_dir = os.path.join(work_dir,'jsons/valid')
	output_dir = os.path.join(work_dir,'coco_json/valid')
	labels = os.path.join(work_dir,'labels.txt')

	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)

	try:
		import pycocotools.mask
	except ImportError:
		print('Please install pycocotools:\n\n    pip install pycocotools\n')
		sys.exit(1)


	def main():
		parser = argparse.ArgumentParser(
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
		parser.add_argument('--input_dir', help='input annotated directory',default= input_dir)
		parser.add_argument('--output_dir', help='output dataset directory',default= output_dir)
		parser.add_argument('--labels', help='labels file',default= labels)
		args = parser.parse_args()

		if not osp.exists(os.path.join(args.output_dir,'JPEGImages')):
			os.makedirs(os.path.join(args.output_dir,'JPEGImages'))
			print('Creating dataset:', args.output_dir)

		now = datetime.datetime.now()

		data = dict(
			info=dict(
				description=None,
				url=None,
				version=None,
				year=now.year,
				contributor=None,
				date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
			),
			licenses=[dict(
				url=None,
				id=0,
				name=None,
			)],
			images=[
				# license, url, file_name, height, width, date_captured, id
			],
			type='instances',
			annotations=[
				# segmentation, area, iscrowd, image_id, bbox, category_id, id
			],
			categories=[
				# supercategory, id, name
			],
		)

		class_name_to_id = {}
		for i, line in enumerate(open(args.labels).readlines()):
			class_id = i - 1  # starts with -1
			class_name = line.strip()
			if class_id == -1:
				assert class_name == '__ignore__'
				continue
			class_name_to_id[class_name] = class_id
			data['categories'].append(dict(
				supercategory=None,
				id=class_id,
				name=class_name,
			))

		out_ann_file = osp.join(args.output_dir, 'annotations.json')
		label_files = glob.glob(osp.join(args.input_dir, '*.json'))
		for image_id, label_file in enumerate(label_files):
			print('Generating dataset from:', label_file)
			with open(label_file) as f:
				label_data = json.load(f)

			base = osp.splitext(osp.basename(label_file))[0]
			out_img_file = osp.join(
				args.output_dir, 'JPEGImages', base + '.jpg'
			)

			img_file = osp.join(
				osp.dirname(label_file), label_data['imagePath']
			)
			img = np.asarray(PIL.Image.open(img_file))
			PIL.Image.fromarray(img).save(out_img_file)
			data['images'].append(dict(
				license=0,
				url=None,
				file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
				height=img.shape[0],
				width=img.shape[1],
				date_captured=None,
				id=image_id,
			))

			masks = {}                                     # for area
			segmentations = collections.defaultdict(list)  # for segmentation
			for shape in label_data['shapes']:
				points = shape['points']
				label = shape['label']
				shape_type = shape.get('shape_type', None)
				mask = labelme.utils.shape_to_mask(
					img.shape[:2], points, shape_type
				)

				if label in masks:
					masks[label] = masks[label] | mask
				else:
					masks[label] = mask

				points = np.asarray(points).flatten().tolist()
				segmentations[label].append(points)

			for label, mask in masks.items():
				cls_name = label.split('-')[0]
				if cls_name not in class_name_to_id:
					continue
				cls_id = class_name_to_id[cls_name]

				mask = np.asfortranarray(mask.astype(np.uint8))
				mask = pycocotools.mask.encode(mask)
				area = float(pycocotools.mask.area(mask))
				bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

				data['annotations'].append(dict(
					id=len(data['annotations']),
					image_id=image_id,
					category_id=cls_id,
					segmentation=segmentations[label],
					area=area,
					bbox=bbox,
					iscrowd=0,
				))

		with open(out_ann_file, 'w') as f:
			json.dump(data, f)


	if __name__ == '__main__':
		main()
