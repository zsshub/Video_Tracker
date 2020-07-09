import argparse
import os
import cv2 as cv
import sys
import numpy as np
import matlab as mlb
import tensorflow as tf
import random as rd
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFont
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw


sys.path.insert(0,"object_detection/research")
sys.path.insert(0,"object_detection/research/object_detection")
from utils import visualization_utils as vis_util
from utils import label_map_util


video_path="data/Cap02t3.avi"
NUM_CLASSES = 2
PATH_TO_CKPT = os.path.join('object_detection/data/exported/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('object_detection/data/labels_items.txt')
#检测间隔
detection_interval = 10
#最大欧氏距离
max_dist_euclidean = 20

def create_graph():
    '''
    建立和读取初始化一个目标检测计算图
    '''
    #定义计算图
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
            serialized_graph = fid.read()
            #定义了文件句柄
            od_graph_def.ParseFromString(serialized_graph)
            #将Graph中的默认参数导入当前计算图。
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
    
def create_label_map():
    '''
    读取标记信息
    '''
    #读取label_map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #定义分类ID和类别
    categories = label_map_util.convert_label_map_to_categories(
        label_map,NUM_CLASSES)
    #创建COCO数据集模式编号
    category_index = label_map_util.create_category_index(categories)
    return category_index

def vid_obj_detection(sess, frame, vid_wight, vid_height, tensor_list, category_index):
    '''
    对提供的图像做目标检测
    Args:
        sess : 运算session
        frame : 检测物体所在的图像
        vid_wight : reshape宽度
        vid_height : reshape高度
        tensor_list : 张量列表
        category_index : 读取标记信息

    Returns:
        detection_frame 标记好检测框的图像数据
        detection_obj_list 检测目标的参数信息列表，包括 box(坐标信息)，class_name(类别)，scores(分数)
        
    '''
    image_np = np.array(frame.copy()).reshape(vid_height,vid_wight, 3).astype(np.uint8)
    image_np_expanded = np.expand_dims(image_np, axis = 0)
    (tensor_boxes, scores, classes, num) = sess.run(
        [tensor_list['detection_boxes'], tensor_list['detection_scores'], tensor_list['detection_classes'], tensor_list['num_detections']],
        feed_dict = {tensor_list['image_tensor']:image_np_expanded})

    #在检测框上画图，返回目标框坐标
    detection_frame,detection_obj_list = vis_util.visualize_boxes_and_labels_on_image_array(
    frame.copy(),
    np.squeeze(tensor_boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates = True,
    line_thickness =1)


    #for i in range(boxes):
    cv.imshow("detection", detection_frame)
    return detection_frame, detection_obj_list

def create_append_tracker(tracker_array,frame,box,obj_class,tracker_id):
    '''
    建立一个新的跟踪器并且将跟踪器添加进跟踪器列表
    Args:
        tracker_array :跟踪器数组
        frame :跟踪目标所在的帧
        box : 跟踪目标的坐标 (left,top,width,height)
        obj_class : 跟踪目标的所属分类
        tracker_id : 新建跟踪器的ID

    Returns:
        返回跟踪器是否创建成功

        跟踪器字典内容
    tracker_dict ={  
        tracker 跟踪器主体
        tracker_objbox跟踪器目标坐标  
        tracker_id #跟踪器ID  
        tracker_color#跟踪器拥有的颜色  
        tracker_class #跟踪器跟踪物体所属分类}
    '''    
    
    tracker_temp = cv.TrackerKCF_create()
    isinit = tracker_temp.init(frame, box)
    if isinit:
        tracker_color = (rd.randint(0,255),rd.randint(0,255),rd.randint(0,255))      
        pt = (int(box[0]+box[2]/2), int(box[1]+box[3]/2))
        tracker_dict = {
            'tracker':tracker_temp,
            'tracker_objbox':box,
            'tracker_id':tracker_id, 
            'tracker_class': obj_class,
            'tracker_color':tracker_color,  
            'tracker_pt': [pt]
            }
        tracker_array.append(tracker_dict)
        return True
    else:
        return False
        
def draw_tracker_box(tracker_array,frame):
    '''
    在跟踪图像上画出新的跟踪框
    Args:
        tracker_array: 跟踪器数组
        frame : 作画的frame
    '''
    for i in range(len(tracker_array)):
        box = tracker_array[i]['tracker_objbox']
        p1 = (int(box[0]),int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        textstr = str(tracker_array[i]['tracker_class'])+" ID:"+str(tracker_array[i]['tracker_id'])
        text_size = cv.getTextSize(textstr,cv.FONT_HERSHEY_DUPLEX,1,1)
        text_width, text_height = text_size[0]
        p3 = (int(p1[0]),int(p1[1]-text_height-2))
        p4 = (int(p1[0]+text_width),int(p1[1]))
        cv.rectangle(frame, p1, p2, tracker_array[i]['tracker_color'], 1, 1)
        cv.rectangle(frame, p3, p4, tracker_array[i]['tracker_color'],-1) 
        cv.putText(frame, textstr, (p1[0],p1[1]-2),cv.FONT_HERSHEY_DUPLEX, 1, (0,0,0),1)
        for j in range(len(tracker_array[i]['tracker_pt'])-1):
            pt1 = tracker_array[i]['tracker_pt'][j]
            pt2 = tracker_array[i]['tracker_pt'][j+1]
            cv.line(frame, pt1,pt2,tracker_array[i]['tracker_color'],2)
    return frame

def tracker_box_updata(tracker_array,tracker_id,newbox):
    '''
    对跟踪器跟踪的目标位置信息进行更新保存
    Args：
        tracker_array : 跟踪器数组列表
        tracker_id：跟踪器Id
        newbox：
    '''
    pt = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))
    tracker_array[tracker_id]['tracker_objbox'] = newbox
    tracker_array[tracker_id]['tracker_pt'].append(pt)
    
def tracker_reinit(tracker_array,tracker_id, frame,newbox):
    '''
    对已有的跟踪器重新初始化，更新检测目标信息
    Args:
        tracker_array : 跟踪器数组
        tracker_id :跟踪器下标id
        frame : 目标所在帧
        newbox: 新的目标框
    '''
    pt = (int(newbox[0]+newbox[2]/2), int(newbox[1]+newbox[3]/2))
    tracker_array[tracker_id]['tracker'] = cv.TrackerKCF_create()
    isinit = tracker_array[tracker_id]['tracker'].init(frame,tuple(newbox))
    tracker_array[tracker_id]['tracker_objbox'] = newbox
    tracker_array[tracker_id]['tracker_pt'].append(pt)
    return isinit 

def cut_frame(frame, last_frame_box ,addper = 0.02):
    '''
    对原始图像进行裁剪，送入网络
    Args:
        frame :原始图像
        last_frame_box :最后一次跟踪正确的box
        addper :坐标外扩比例，以全局图片大小为参照
    '''
    #y0,y1,x0,x1
    wight = frame.shape[1]
    height = frame.shape[0]
    margin_wight = 0.02*frame.shape[1]
    margin_height = 0.02*frame.shape[0]
    y0 = int(last_frame_box[1]-margin_height)
    y1 = int(last_frame_box[1]+last_frame_box[3]+margin_height)
    x0 = int(last_frame_box[0]-margin_wight)
    x1 = int(last_frame_box[0] + last_frame_box[2]+margin_wight)
    if y0 < 0 :
        y0 = 0
    if y1>height:
        y1 = height
    if x0 <0:
        x0 = 0
    if x1 > wight:
        x1 = wight

    temp_box = (x0,y0,x1-x0,y1-y0)
    cut_frame = frame[y0:y1, x0:x1]
    return cut_frame, temp_box

def change_basis_point(detection_list, local_box_in_global):
    local_wight = local_box_in_global[2]
    local_height = local_box_in_global[3]
    for i in range(len(detection_list)):
        box = list(detection_list[i]['box'])
        x0 = box[0]/local_wight + local_box_in_global[0]
        y0 = box[1]/local_height + local_box_in_global[1]
        w = box[2]
        h = box[3]
        detection_list[i]['box'] = (x0,y0,w,h)

    return detection_list
      

def main():    
    #读取视频文件
    cap = cv.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print("视频打开错误")
    #     sys.exit()

    
    #建立初始化计算图
    detection_graph = create_graph()
    #读取标记信息
    category_index = create_label_map()

    #取得视频格式信息
    vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_wight = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    
    Tracker_array = []#跟踪器数组 


    with detection_graph.as_default():
         with tf.device('/device:GPU:0'):
            with tf.Session(graph = detection_graph,config=tf.ConfigProto(log_device_placement=True))as sess:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                tensor_list ={'image_tensor':image_tensor,'detection_boxes':detection_boxes, 
                         'detection_scores':detection_scores, 'detection_classes':detection_classes, 'num_detections':num_detections}

                #初始化当前帧
                now_frame = -1
                tracker_top_id = 0
                #cap.set(cv.CAP_PROP_POS_FRAMES, 849)
                while(1):
                #读取下一帧
                    _, frame = cap.read()
                    now_frame += 1
                    print(now_frame)
                    if frame is None:
                        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                        now_frame = -1
                        _, frame = cap.read()
                        Tracker_array.clear()
                        tracker_top_id = 0
                    detection_frame = frame.copy()
                    tracker_frame = frame.copy()
                    #循环播放
                    
    

                    #每经过固定间隔后将检测结果反馈给跟踪模块
                    if(now_frame%detection_interval == 0):

                        #对当前帧进行目标检测，并返回标有检测框的图像，坐标，类别参数
                        detection_frame, tracker_new_list = vid_obj_detection(
                            sess, frame, vid_wight, vid_height, tensor_list, category_index)

                        #没有跟踪器在运行的情况下，对所有目标都建立跟踪器

                        if len(Tracker_array) == 0:
                            for i in range(len(tracker_new_list)):     
                                tracker_top_id += 1
                                #建立并初始化跟踪器
                                create_append_tracker(Tracker_array, frame, tracker_new_list[i]['box'], tracker_new_list[i]['class_name'], tracker_top_id) 
                        
                        #当前有跟踪器在运行的情况下，执行跟踪器，并对目标和跟踪器间做一个匹配
                        else:
                            #对已有的跟踪器做匹配更新
                            del_Tracker_array_array = []
                            for i in range(len(Tracker_array)):
                                tracker_dict_temp =Tracker_array[i]
                                isok, tracker_newbox = tracker_dict_temp["tracker"].update(frame)
                                if not isok:
                                    tracker_newbox = Tracker_array[i]['tracker_objbox']
                                dist_dict_temp={}
                                for j in range(len(tracker_new_list)):
                                    if tracker_new_list[j]['class_name'] == tracker_dict_temp['tracker_class']:                                           
                                        dist_euclidean =np.linalg.norm(np.array(tracker_newbox)-np.array(tracker_new_list[j]['box']))
                                        #保存第一次计算的欧式距离
                                        if len(dist_dict_temp)==0:
                                            dist_dict_temp={'dist':dist_euclidean, 'detection_id':j}
                                        else:
                                            #当前双方的欧氏距离比保存了的还小，则更新
                                            if dist_euclidean < dist_dict_temp['dist']:
                                                dist_dict_temp={'dist':dist_euclidean, 'detection_id':j}
                                if isok:    
                                    if len(dist_dict_temp) != 0 :
                                        #将跟跟踪器最匹配，且欧式距离小于阈值的目标框坐标更新到跟踪器，并删除该目标框
                                        tracker_reinit(Tracker_array,i, frame, tracker_new_list[dist_dict_temp['detection_id']]['box'])
                                        del tracker_new_list[dist_dict_temp['detection_id']]

                                else:
                                    if len(dist_dict_temp) != 0 :
                                        #将跟跟踪器最匹配，且欧式距离小于阈值的目标框坐标更新到跟踪器，并删除该目标框
                                        tracker_reinit(Tracker_array,i, frame, tracker_new_list[dist_dict_temp['detection_id']]['box'])
                                        del tracker_new_list[dist_dict_temp['detection_id']]
                                    else:
                                        del_Tracker_array_array.append(i)
                            for i in del_Tracker_array_array:
                                del Tracker_array[i]
                                                                  
                            #对剩下的多余的目标新建跟踪器
                            for i in range(len(tracker_new_list)):
                                tracker_top_id += 1
                                create_append_tracker(Tracker_array,frame,tracker_new_list[i]['box'],tracker_new_list[i]['class_name'],tracker_top_id)
                        
                        cv.imshow("detection",detection_frame)

                    #跟踪器工作
                    
                    else:
                        del_Tracker_array_array = []
                        for i in range(len(Tracker_array)):
                            tracker_dict_temp = Tracker_array[i]
                            is_tracker, tracker_newbox = tracker_dict_temp["tracker"].update(frame)
                            #正常跟踪
                            if is_tracker:
                                tracker_box_updata(Tracker_array,i,tracker_newbox)
                            #跟踪失败,需要对该区域进行检测
                            else:
                                #对丢失区域进行裁剪 y0:y1,x0:x1
                                last_frame_box = Tracker_array[i]['tracker_objbox']
                                cut_detection_frame, last_add_box= cut_frame(frame, last_frame_box,0.02)
                                cv.imshow("cut_frame",cut_detection_frame)
                                #对丢失区域重新进行检测
                                try:
                                    _, detection_newobj_list = vid_obj_detection(
                                            sess, cut_detection_frame, cut_detection_frame.shape[1], cut_detection_frame.shape[0], tensor_list, category_index) 
                                    detection_newobj_list = change_basis_point(detection_newobj_list, last_add_box)
     
                                except:
                                    detection_newobj_list = []
                                if len(detection_newobj_list)==0:
                                    #没有检测到目标，说明物体已经离开视线，删除跟踪器
                                    del_Tracker_array_array.append(i)
                                elif len(detection_newobj_list)==1:                                   
                                    if detection_newobj_list[0]['class_name']==tracker_dict_temp['tracker_class']:
                                        #只有一个目标，且类别相同，更新跟踪器
                                        tracker_reinit(Tracker_array, i, frame, detection_newobj_list[0]['box'])
                                    else:
                                        #只有一个目标，但类别不同。删除原来的跟踪器，新建跟踪器
                                        del_Tracker_array_array.append(i)
                                        tracker_top_id +=1
                                        create_append_tracker(Tracker_array, frame, detection_newobj_list[0]['box'], detection_newobj_list[0]['class_name'],tracker_top_id)  
                                else:
                                    #检测结果多于一个，说明有重叠的物体出现
                                    dist_euclidean_dict={}#保存欧式距离最小的组合
                                   
                                    del_detection_newobj_list_array = []#保存要删除的检测框下标
                                    for j in range(len(detection_newobj_list)):
                                        if tracker_dict_temp['tracker_class']==detection_newobj_list[j]['class_name']:
                                            #类别相同的类需要进行匹配
                                            dist_euclidean =np.linalg.norm(np.array(tracker_newbox)-np.array(detection_newobj_list[j]['box']))
                                            if len(dist_euclidean_dict) == 0:
                                                dist_euclidean_dict ={'detection_box_id':j ,'dist_euclidean':dist_euclidean}
                                            elif dist_euclidean < dist_euclidean_dict['dist_euclidean']:
                                                dist_euclidean_dict ={'detection_box_id':j ,'dist_euclidean':dist_euclidean}
                                        
                                        else:
                                            #类别不同的直接新建跟踪器,并在检测列表中删除该目标
                                            tracker_top_id += 1
                                            create_append_tracker(Tracker_array, frame, detection_newobj_list[j]['box'], detection_newobj_list[j]['class_name'],tracker_top_id)
                                            del_detection_newobj_list_array.append(j)
                                    #删除检测框，避免下标溢出的情况       
                                    
                                    if len(dist_euclidean_dict)!=0:
                                        #字典不为0，说明有目标与当前跟踪器匹配，需要更新跟踪器
                                        tracker_reinit(Tracker_array,i,frame,detection_newobj_list[dist_euclidean_dict['detection_box_id']]['box'])
                                        del detection_newobj_list[dist_euclidean_dict['detection_box_id']]

                                    if len(detection_newobj_list)!=0:
                                        #如果还有剩下的目标，全部新建跟踪器
                                        for k in range(len(detection_newobj_list)):
                                            tracker_top_id += 1
                                            create_append_tracker(Tracker_array,frame,detection_newobj_list[k]['box'],detection_newobj_list[k]['class_name'],tracker_top_id)
                                    for k in del_detection_newobj_list_array:
                                        del detection_newobj_list[k]
                        try:
                            for k in del_Tracker_array_array:
                                del Tracker_array[k]
                        except:
                            continue

                        

                    #显示结果                 
                    
                    draw_tracker_box(Tracker_array,tracker_frame)
                    #cv.imshow("tracker",tracker_frame)
                    cv.imshow("source",frame)
                    cv.imshow("tracker",tracker_frame)
                    key = cv.waitKey(24)
                    if key == 27:
                        break

     
if __name__ == '__main__':
    main()

