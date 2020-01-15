import json
import glob
import os
import PIL.Image
from labelme import utils

work_dir = '/media/job/myjob/new表计/new_orign/fangdianjishuqi_yuanbiao/segment/jsons'

json_list = glob.glob(os.path.join(work_dir,'*.json'))

for file_json in json_list:
	data = json.load(open(os.path.join(work_dir,file_json)))
	img = utils.img_b64_to_arr(data['imageData'])
	img_name = os.path.join(work_dir,file_json.replace('.json','.jpg'))
	PIL.Image.fromarray(img).save(img_name)