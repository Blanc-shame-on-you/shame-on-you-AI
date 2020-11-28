import sqlite3
import numpy as np
import io
import cv2

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

    con = sqlite3.connect(".info.db", detect_types=sqlite3.PARSE_DECLTYPES)
    cur = con.cursor()
    #cur.execute("create table test (arr array)")
    return cur,con

def crop_and_save(image,location,cur,con):
    #xmin, ymin, xmax, ymax=location 원본
    xmin, ymax, xmax, ymin=location
    print("xmin"+str(xmin))
    print("ymin"+str(ymin))
    print("xmax"+str(xmax))
    print("ymax"+str(ymax))
    croped_image=image[ymin:ymax,xmin:xmax  ,::-1] 
    cv2.imshow("32523",croped_image)
    print('insert')
    cur.execute("insert into people (arr) values (?)", (croped_image, ))
    con.commit()
    '''except: 
        print('create')
        cur.execute("create table people (arr array)")
        con.commit()'''
def test_save(image,cur,con):
    cur.execute("insert into people (arr) values (?)", (image, ))
    con.commit()
def load(cur):
    cur.execute("select arr from people")
    #print(cur.fetchall())
    data = cur.fetchall()
    return data

'''cursor=setting()
img=cv2.imread('C:/Users/Lee/Pictures/gan_1.jpg')
test_save(img,cursor)
cv2.imshow("test",load(cursor))'''