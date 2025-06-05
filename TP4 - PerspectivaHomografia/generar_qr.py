import qrcode

# Contenido del QR
data = "test"

# Crear el objeto QR
qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4
)
qr.add_data(data)
qr.make(fit=True)

# Generar la imagen
img = qr.make_image(fill_color="black", back_color="white")

# Guardar a archivo
img.save("qr_test.png")
print("QR generado como qr_test.png")