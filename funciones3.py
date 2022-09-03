cadena="hola mundo"
formato_izquierda=format(cadena,"<100")
formato_derecha=format(cadena,">100")
formato_central=format(cadena,"^100")

print("formato izq==> \n",formato_izquierda)
print("formato cen==> \n",formato_central)
print("formato der==> \n",formato_derecha)