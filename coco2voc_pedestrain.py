from pycocotools.coco import COCO
import shutil
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString


# 获取所需要的类名和id
# path为类名和id的对应关系列表的地址（标注文件中可能有很多类，我们只加载该path指向文件中的类）
# 返回值是一个字典，键名是类名，键值是id
def get_classes_and_index(path):
	D = {}
	f = open(path)
	for line in f:
		temp = line.rstrip().split(',', 2)
		print("temp[0]:" + temp[0] + "\n")
		print("temp[1]:" + temp[1] + "\n")
		D[temp[1]] = temp[0]
	return D


def make_voc_dir():
	# labels 目录若不存在，创建labels目录。若存在，则清空目录
	if not os.path.exists('../VOC2020/Annotations'):
		os.makedirs('../VOC2020/Annotations')
	if not os.path.exists('../VOC2020/ImageSets'):
		os.makedirs('../VOC2020/ImageSets')
		os.makedirs('../VOC2020/ImageSets/Main')
	if not os.path.exists('../VOC2007/JPEGImages'):
		os.makedirs('../VOC2020/JPEGImages')


if __name__ == '__main__':
	make_voc_dir()
	VOCRoot = '../VOC2020'
	dataDir = '/media/data/dataset/COCO/data'  # COCO数据集所在的路径
	dataType = 'coco2017'  # 要转换的COCO数据集的子集名
	subDataType = 'train2017'
	annFile = '%s/%s/annotations/instances_%s.json' % (dataDir, dataType,subDataType)  # COCO数据集的标注文件路径
	classes = get_classes_and_index('/media/data/dataset/COCO/coco_list.txt')  # {'person': '0'}

	coco = COCO(annFile)  # 加载解析标注文件

	imgIds = coco.getImgIds()  # 获取标注文件中所有图片的COCO Img ID
	catIds = coco.getCatIds()  # 获取标注文件总所有的物体类别的COCO Cat ID

	for imgId in imgIds:
		objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
		print('imgId :%s' % imgId)
		Img = coco.loadImgs(imgId)[0]  # 加载图片信息
		print('Img :%s' % Img)
		filename = Img['file_name']  # 获取图片名   '000000391895.jpg'
		width = Img['width']  # 获取图片尺寸
		height = Img['height']  # 获取图片尺寸 360
		print('filename :%s, width :%s ,height :%s' % (filename, width, height))
		annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)  # 获取该图片对应的所有COCO物体类别标注ID
		print('annIds :%s' % annIds)

		node_root = Element('annotation')
		node_folder = SubElement(node_root, 'folder')
		node_folder.text = 'JPEGImages'
		node_filename = SubElement(node_root, 'filename')
		node_filename.text = 'VOC2007/JPEGImages/%s' % filename
		node_size = SubElement(node_root, 'size')
		node_width = SubElement(node_size, 'width')
		node_width.text = '%s' % width
		node_height = SubElement(node_size, 'height')
		node_height.text = '%s' % height
		node_depth = SubElement(node_size, 'depth')
		node_depth.text = '3'

		for annId in annIds:
			anns = coco.loadAnns(annId)[0]  # 加载标注信息
			catId = anns['category_id']  # 获取该标注对应的物体类别的COCO Cat ID 1
			cat = coco.loadCats(catId)[0]['name']  # 获取该COCO Cat ID对应的物体种类名 'person'

			# 如果该类名在我们需要的物体种类列表中，将标注文件转换为YOLO需要的格式
			if cat in classes:
				objCount = objCount + 1

				cls_id = classes[cat]  # 0
				box = anns['bbox']  # <class 'list'>: [339.88, 22.16, 153.88, 300.73]
				size = [width, height]
				xmin = int(box[0]) + 1
				ymin = int(box[1]) + 1
				obj_width = int(box[2])
				obj_height = int(box[3])
				xmax = xmin + obj_width
				ymax = ymin + obj_height

				difficult = 0
				if obj_height <= 4 or obj_width <= 4:
					difficult = 1

				node_object = SubElement(node_root, 'object')
				node_name = SubElement(node_object, 'name')
				node_name.text = cat
				node_difficult = SubElement(node_object, 'difficult')
				node_difficult.text = '%s' % difficult
				node_bndbox = SubElement(node_object, 'bndbox')
				node_xmin = SubElement(node_bndbox, 'xmin')
				node_xmin.text = '%s' % xmin
				node_ymin = SubElement(node_bndbox, 'ymin')
				node_ymin.text = '%s' % ymin
				node_xmax = SubElement(node_bndbox, 'xmax')
				node_xmax.text = '%s' % xmax
				node_ymax = SubElement(node_bndbox, 'ymax')
				node_ymax.text = '%s' % ymax
				node_name = SubElement(node_object, 'pose')
				node_name.text = 'Unspecified'
				node_name = SubElement(node_object, 'truncated')
				node_name.text = '0'

		if objCount > 0:
			image_path = VOCRoot + '/JPEGImages/' + filename
			xml = tostring(node_root, pretty_print=True)  # 'annotation'
			dom = parseString(xml)
			xml_name = filename.replace('.jpg', '.xml')
			xml_path = VOCRoot + '/Annotations/' + xml_name
			with open(xml_path, 'wb') as f:
				f.write(xml)
			shutil.copy('%s/%s/%s/%s' % (dataDir, dataType, subDataType,filename), '../VOC2020/JPEGImages/' + filename)