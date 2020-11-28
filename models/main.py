import platform
import sys
model_path = ['/home/gpuadmin/shame-on-you/models/mask_detect/', 'D:/project/shame-on-you-AI/models/mask_detect/']
def choose_path():
    if platform.system() == 'Windows':
        return model_path[1]
    elif platform.system() == 'Linux':
        return model_path[0]


sys.path.append(choose_path())
from tensorflow_mask_detect import inference
#from pytorch_infer import inference
import time
import cv2
import face_recognition
import sqlite3
import numpy as np
import io
import cv2
import time
import numpy as np
import db
#resullt 마스크 안쓴 사람 수 
#locations 마스크 안쓴사람의 얼굴 좌표 [[[xmin, ymin, xmax, ymax]]
#base info {mask = T/F, time, before_time}

def detect(img_raw, known_face_encodings,base_info,process_this_frame = True):
    #temp_img=img_raw.copy()
    cursor,cor=db.setting()
    result, locations =inference(img_raw.copy())
    #img_raw에 cycle gan 적용 그리고 이미지 늘려서 거기다가 이미지 크롭
    face_locations = []
    for xmin, ymin, xmax, ymax in locations:
        img_raw = cv2.rectangle(img_raw,(xmin,ymin),(xmax,ymax),(255,0,0),2)
        #face_img = img_raw[ymin:ymax,xmin:xmax  ,::-1]
        #결과값이 잘안나오면 아래 줄 없애고 face location 찾는 줄 추가
        '''temp = [int(ymax/4),int(xmax/4),int(ymin/4),int(xmin/4)]
        face_locations.append(temp)'''
    print(result)
    if len(result)!=0:
        print('enter')
        #small_frame = cv2.resize(img_raw, (0, 0), fx=0.25, fy=0.25)    
        #rgb_small_frame = small_frame[:, :, ::-1]
        img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        show_flag = np.zeros(len(face_locations))
        print("face_locations"+str(len(face_locations)))
        if process_this_frame:
            face_encodings = face_recognition.face_encodings(img, face_locations)
            for index,face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if True in matches:
                    print('True!')
                    first = matches.index(True)
                    show_flag[first] = 1
                    print(index)
                    try:
                        if result[index] == 1:
                            if base_info[first]['mask']:
                                #base_info[first]['time'] = 0
                                base_info[first]['before_time'] = time.time()
                            else:
                                base_info[first]['time'] += time.time() - base_info[first]['before_time']
                            base_info[first]['mask'] = False
                        else:
                            base_info[first]['mask'] = True
                    except:
                        pass    
                else:#내일 여기 face location 거꾸로 해서 주던가 아니면 db파일에서 순서를 바꾸던가 해서 해결하기
                    print('False!')
                    '''print(face_locations)
                    #face_locations[0].reverse()
                    print(face_locations[0][::-1])
                    print("ds;alfkj;dfkjsaksj;afjsdf;lkj")'''
                    known_face_encodings.append(face_encoding)
                    db.crop_and_save(img_raw,face_locations[0][::-1],cursor,cor)
                    base_info.append({"mask" : True,"time" : 0,"before_time" : 0})
            for index,flag in enumerate(show_flag):
                if flag == 0:
                    base_info[index]['mask'] = True
        print(show_flag)
        print("사람 수"+str(len(known_face_encodings)))
        print(base_info)
        #face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

    return img_raw,known_face_encodings,base_info


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def setting():
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)

    x = np.arange(12).reshape(2,6)

    con = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    cur.execute("create table test (arr array)")
    return cur

def crop_and_save(image,ymin,ymax,xmin,xmax,cur):
    croped_image=image[ymin:ymax,xmin:xmax  ,::-1]
    cur.execute("insert into test (arr) values (?)", (croped_image, ))

def test_save(image,cur):
    cur.execute("insert into test (arr) values (?)", (image, ))

def load(cur):
    cur.execute("select arr from test")
    data = cur.fetchone()[0]
    return data


if __name__ == "__main__":
    known_face_encodings = []
    base_info = []
    akt = 1
    cap = cv2.VideoCapture(0)
    process_frame = True
    while cap.isOpened():
        _,img_raw = cap.read()
        img ,known_face_encodings,base_info = detect(img_raw, known_face_encodings, base_info, process_frame)
        process_frame = not process_frame
        cv2.imshow("img",img)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


'''
cursor=setting()
img=cv2.imread('C:/Users/Lee/Pictures/gan_1.jpg')
test_save(img,cursor)   
cv2.imshow("test",load(cursor))
cv2.waitKey(0)
'''