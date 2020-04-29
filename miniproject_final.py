import cv2
import math
import numpy as np
import time

cap = cv2.VideoCapture('/home/philip/Desktop/OpenCV/mptools.mp4')


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))


def scissorTest():
    for x in range(0,no_contours):
        length = family_tree[x].__len__() - 1
        if length >= 2 and x in parents:
            parent = x
            tri_area,sizes = cv2.minEnclosingTriangle(contours[parent])
            hull = cv2.convexHull(contours[parent],returnPoints = True)
            poly = cv2.approxPolyDP(hull, 0.001, True)
            hull_area = cv2.contourArea(poly)
            ratio = hull_area/tri_area
            if ratio > 0.85:
                cv2.drawContours(ref, contours, parent, (0,0,255), 2)
                M = moments[x]
                text_x = int(M['m10']/M['m00'])
                text_y = int(M['m01']/M['m00'])
                cv2.putText(ref, "Scissor",(text_x,text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255), 2)


def thermoTest():
    for x in range(0,no_contours):
        length = family_tree[x].__len__() - 1
        if length >= 1 and x in parents:
            for y in range(1,length+1):
                child = family_tree[x][y]
                center,sizes,asd = cv2.minAreaRect(contours[child])
                cnt_area = cv2.contourArea(contours[child], 0)
                width = sizes[0]
                height = sizes[1]
                squareness = cnt_area/(height*width)
                if squareness > 0.9:
                    cv2.drawContours(ref, contours, x, (0,255,0), 2)
                    M = moments[x]
                    text_x = int(M['m10']/M['m00'])
                    text_y = int(M['m01']/M['m00'])
                    cv2.putText(ref, "Thermometer",(text_x,text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)


def spoonTest():
    for x in range(0,no_contours):
        length = family_tree[x].__len__() - 1
        if length <= 1 and x in parents:
            M = moments[x]
            center,sizes,asd = cv2.minAreaRect(contours[x])
            width = sizes[0]
            height = sizes[1]
            com_x = int(M['m10']/M['m00'])
            com_y = int(M['m01']/M['m00'])
            com = (com_x,com_y)
            hull = cv2.convexHull(contours[x],returnPoints = False)
            defects = cv2.convexityDefects(contours[x], hull)
            cnt = contours[x]
            counter = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                far_x = far[0]
                far_y = far[1]
                x_dist = abs(com_x - far_x)
                y_dist = abs(com_y - far_y)
                euc_dist = math.sqrt((x_dist*x_dist)+(y_dist*y_dist))
                ratio = euc_dist/height
                if ratio < 0.1:
                    counter = counter + 1
                if counter >= 2:
                    cv2.drawContours(ref, contours, x, (255,0,0), 2)
                    cv2.putText(ref, "Spoon",(com_x,com_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)


def spatulaTest():
    for x in range(0,no_contours):
        length = family_tree[x].__len__() - 1
        if length <= 1 and x in parents:
            rect = cv2.minAreaRect(contours[x])
            width = int(round(rect[1][0],0))
            height = int(round(rect[1][1],0))
            half_width = int(round(width/2,0))
            half_height = int(round(height/2,0))
            t_half = ((rect[0][0],rect[0][1]-(rect[1][1]/4)), (width,(half_height)), (rect[2]))

            t_corner = ((rect[0][0]-half_width), (rect[0][1]-half_height))
            t_corner_x = int(round(t_corner[0],0))
            t_corner_y = int(round(t_corner[1],0))

            b_half = ((rect[0][0],rect[0][1]+(rect[1][1]/4)), (width,(half_height)), (rect[2]))
            b_corner = ((rect[0][0]-half_width), (rect[0][1]))
            b_corner_x = int(round(b_corner[0],0))
            b_corner_y = int(round(b_corner[1],0))

            top_mask = np.zeros(opening.shape[:2],np.uint8)
            top_mask[t_corner_y:t_corner_y+half_height,t_corner_x:t_corner_x+width] = 255
            top_img = cv2.bitwise_and(opening,opening,mask = top_mask)
            _, top_contours,_ = cv2.findContours(top_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            top_area = cv2.contourArea(top_contours[-1], 0)
            top_ratio = top_area/(half_height*width)

            bot_mask = np.zeros(opening.shape[:2],np.uint8)
            bot_mask[b_corner_y:b_corner_y+half_height,b_corner_x:b_corner_x+width] = 255
            bot_img = cv2.bitwise_and(opening,opening,mask = bot_mask)
            _, bot_contours,_ = cv2.findContours(bot_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            bot_area = cv2.contourArea(bot_contours[-1], 0)
            bot_ratio = bot_area/(half_height*width)

            if top_ratio > 0.7 and bot_ratio < 0.2:
                cv2.drawContours(ref, contours, x, (255,255,0), 2)
                M = moments[x]
                text_x = int(M['m10']/M['m00'])
                text_y = int(M['m01']/M['m00'])
                cv2.putText(ref, "Spatula",(text_x,text_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,0), 2)


excp_cnt = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    src = frame
    ref = frame

    mask1 = cv2.inRange(src,(0,0,0),(75,85,85))
    mask2 = cv2.inRange(src,(165,145,145),(255,255,255))
    mask = mask1 | mask2

    closing = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel)
    #cv2.imshow('Boundary',opening)

    image, contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    no_contours = contours.__len__()

    #MOMENT LIST
    moments = []
    for x in range (0,no_contours):
        moments.append(cv2.moments(contours[x]))

    #CONTOUR FAMILY TREE
    parents = []
    childs = []
    family_tree = []

    for x in range(0,no_contours):
        family_tree.append([x])

    for x in range(0,no_contours):
        if hierarchy[0][x][3] > -1:
            parent = hierarchy[0][x][3]
            child = x
            parents.append(parent)
            childs.append(child)
            family_tree[parent].append(child)

    #CONTOUR TESTS
    scissorTest()

    thermoTest()

    spoonTest()

    try :
        spatulaTest()
    except IndexError:
        None

    cv2.imshow('Contours',ref)

    k = cv2.waitKey(5) & 0xFF                                                   #Press escape to exit
    if(k == 27):
        break

cap.release()
cv2.destroyAllWindows()
