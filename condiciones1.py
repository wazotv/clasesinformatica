
#06-08-2022

#pida la a un usuario a su nombre su edad, determine si es mayor de edad, 
#y muestra a un mensaje en pantalla diciendo:
nombre=input("ingrese su nombre")
edad=int(input("ingrese la edad--->"))

if 18<=edad<=150:
    print("{},es mayor de edad".format(nombre))
elif 0<edad<18:
    print(nombre,"es menor de edad")




