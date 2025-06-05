import cv2 as cv
import numpy as np
import os

# Estado global
homography = None
click_points = []
output_counter = 1
last_warped = None
modo_manual = False  # 🔄 Nuevo estado para modo manual

def ordenar_puntos(puntos):
    puntos = puntos.reshape(4, 2)
    suma = puntos.sum(axis=1)
    resta = np.diff(puntos, axis=1)

    ordenados = np.zeros((4, 2), dtype=np.float32)
    ordenados[0] = puntos[np.argmin(suma)]
    ordenados[2] = puntos[np.argmax(suma)]
    ordenados[1] = puntos[np.argmin(resta)]
    ordenados[3] = puntos[np.argmax(resta)]

    return ordenados

def detect_qr(frame):
    detector = cv.QRCodeDetector()
    retval, points = detector.detect(frame)
    if retval and points is not None:
        puntos_ordenados = ordenar_puntos(points[0])
        dst_pts = np.array([[0,0], [400,0], [400,400], [0,400]], dtype=np.float32)
        return cv.findHomography(puntos_ordenados, dst_pts)
    return None, None

def mouse_click(event, x, y, flags, param):
    global click_points
    if modo_manual and len(click_points) < 4 and event == cv.EVENT_LBUTTONDOWN:
        click_points.append((x, y))

def draw_grid(frame, H, size=(400, 400), rows=3, cols=3):
    for i in range(rows + 1):
        pt1 = np.array([[0, i * size[1] // rows]], dtype='float32')
        pt2 = np.array([[size[0], i * size[1] // rows]], dtype='float32')
        pt1 = cv.perspectiveTransform(np.array([pt1]), np.linalg.inv(H))[0][0]
        pt2 = cv.perspectiveTransform(np.array([pt2]), np.linalg.inv(H))[0][0]
        cv.line(frame, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)
    for j in range(cols + 1):
        pt1 = np.array([[j * size[0] // cols, 0]], dtype='float32')
        pt2 = np.array([[j * size[0] // cols, size[1]]], dtype='float32')
        pt1 = cv.perspectiveTransform(np.array([pt1]), np.linalg.inv(H))[0][0]
        pt2 = cv.perspectiveTransform(np.array([pt2]), np.linalg.inv(H))[0][0]
        cv.line(frame, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0,255,0), 1)

def guardar_imagen(img):
    global output_counter
    os.makedirs("salidas", exist_ok=True)
    filename = f"salidas/vista_frontal_{output_counter:03d}.jpg"
    cv.imwrite(filename, img)
    print(f"📸 Imagen guardada como {filename}")
    output_counter += 1

def main():
    global homography, click_points, last_warped, modo_manual
    cap = cv.VideoCapture(0)
    cv.namedWindow("Webcam")
    cv.setMouseCallback("Webcam", mouse_click)

    print("Presioná 'q' para detectar QR, 'h' para seleccionar 4 puntos manualmente, 's' para guardar, 'Esc' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # 🔄 Si estamos en modo manual y ya hay 4 clics, calculamos homografía y salimos
        if modo_manual and len(click_points) == 4:
            dst_pts = np.array([[0,0], [400,0], [400,400], [0,400]], dtype=np.float32)
            homography, _ = cv.findHomography(np.array(click_points, dtype=np.float32), dst_pts)
            click_points = []
            modo_manual = False
            print("✅ Homografía manual calculada.")

        # Mostrar grilla si hay homografía
        if homography is not None:
            cv.putText(display, "Homografia activa", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            draw_grid(display, homography, size=(400, 400))
            last_warped = cv.warpPerspective(frame, homography, (400, 400))
            cv.imshow("Vista frontal", last_warped)

        # Dibujar puntos seleccionados
        for pt in click_points:
            cv.circle(display, pt, 5, (0, 0, 255), -1)

        cv.imshow("Webcam", display)
        key = cv.waitKey(1) & 0xFF

        if key == ord('q'):
            H, _ = detect_qr(frame)
            if H is not None:
                homography = H
                print("✅ Homografía QR calculada.")
            else:
                print("❌ QR no detectado.")

        elif key == ord('h'):
            click_points = []
            modo_manual = True
            print("🖱️ Hacé clic en 4 puntos...")

        elif key == ord('s'):
            if last_warped is not None:
                guardar_imagen(last_warped)
            else:
                print("⚠️ No hay imagen transformada para guardar.")

        elif key == 27:  # ESC
            print("Saliendo...")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
