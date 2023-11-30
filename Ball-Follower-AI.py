import cv2
import numpy as np

# Cargar el modelo YOLOv3
net = cv2.dnn.readNet('./Lib/yolov3.weights', './Lib/yolov3.cfg')

# Cargar las clases
classes = []
with open('./Lib/coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Inserta el nombre de tu video
cap = cv2.VideoCapture('./Video/1.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        print("El video ha terminado")
        break

    height, width, _ = frame.shape

    # Construir un blob desde la imagen
    blob = cv2.dnn.blobFromImage(frame, 0.5/255.0, (249, 249), swapRB=True, crop=False)

    # Establecer la entrada para la red neuronal
    net.setInput(blob)

    # Obtener las capas de salida
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Procesar las detecciones
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar la supresión no máxima para eliminar las detecciones débiles
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices:
        box = boxes[i]
        x, y, w, h = box

        # Dibujar el cuadro del objeto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 5)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
