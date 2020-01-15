import json
import glob
import os

root_dir = '/media/ygj/ygj/others/tanjun/变压器表计识别/分类样本/new_orign/bianyaqi_fangbiao'
part_dirs = ['coarse_scale_json','red_point_json','precise_scale_json','white_point_json']
output_dir = '/media/ygj/ygj/others/tanjun/变压器表计识别/分类样本/new_orign/bianyaqi_fangbiao/json_all'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_path = part_dirs[0]

tmp_dir = os.path.join(root_dir,start_path)
if os.path.exists(tmp_dir):
    file_jsons = glob.glob(os.path.join(tmp_dir,'*.json'))
    for file_json in file_jsons:
        data_shape = []
        with open(file_json) as fi:
            data = json.load(fi)
            data_shape += data['shapes']

            for path_dir in part_dirs[1:]:
                tmp_file = file_json.replace(start_path,path_dir)
                with open(tmp_file) as fi:
                    data = json.load(fi)
                    data_shape += data['shapes']

            data['shapes'] = data_shape

            output_file = file_json.replace(start_path,'json_all')

            with open(output_file,'w') as fw:
                json.dump(data,fw)

        