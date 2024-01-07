import cv2
import numpy as np

def layer_set(img):
    classes = []
    net = cv2.dnn.readNet(weights_path, config_path)

    with open(names_path, "r") as f:
        classes =[line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    print(outs)

    return img, classes, colors, height, width, channels, outs

def detection(img, classes, colors, height, width, channels, outs):
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = colors[i]
                        cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
                        cv2.putText(img, label, (x, y+30), font, 1, color, 1)
    return img


    
if __name__ == '__main__':
    weights_path = '/home/seok/123/yolov4.weights'
    config_path = '/home/seok/123/yolov4.cfg'
    names_path = '/home/seok/123/coco.names'

    img = cv2.imread('/home/seok/123/person.jpg')

    img, classes, colors, height, width, channels, outs = layer_set(img)
    detect_img = detection(img, classes, colors, height, width, channels, outs)
    cv2.imwrite('detect_out.jpg', detect_img)