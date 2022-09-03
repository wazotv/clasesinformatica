#cadena="hola mundo"
#lista=[1,2,3]
#entero=10
#
#print("\nfuncionalidades para cadena==> ^\n\n",dir(cadena))
#print("\nfuncionalides para listas\n\n",dir(lista))
#print("\nfuncionalidades para entero\n\n",dir(entero))





secuencia=range(1,11) # el inicio se toma pero el final no se toma

print(list (secuencia))

secuencia2=range(20,30) # cuando el salto es uno no es necesario el terce parametro 


print(list(secuencia2))


secuencia3=range(-10,11,2)

print(list(secuencia3))

secuencia4=range(-9,6,3)
print(list(secuencia4))

secuencia5=range(10,-1,-1)

print(list(secuencia5))


secuencia6=range(990,0,-15)

print(list(secuencia6))

#tamaño:len()
#sum()
#min()
#reversa:reversed

lista=[1,2,3,5,8,8,9]
secuencia7=range(1,10000,3)

print("tamaño secuencia",len(secuencia7))
print("tamaño de lista",len(lista))

print("el valor maximo de la secuancia",min(secuencia7))
print("el valor maximo de la lista",max(lista))


print("revertir",list(reversed(secuencia7)))
print("revertir lista",list(reversed(lista)))


#repetir el ejercicio anterior usando reversed




