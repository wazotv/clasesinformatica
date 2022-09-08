#calcule el salario de un empleado teniendo en cuenta las siguinetes condiciones:
#ventas =>[5,20] seguros=> aumento 20%
#salario base 
#ventas [21,50] aumenta 30% sobre la base del salario 

#ventas [51,infinito] seguros aumenta el 35% sobre el salario base 
salario_base=1000000
ventas=int(input("ingrese el numero de ventas que realizo"))
while True:
    try:
        seguros=int(input("ingrese la cantidad de seguros"))
    except ValueError:
        continue
    if seguros <=0:
        print("valor incorrecto")
        print("ingrese otro valor")
    else:
        break

if 5<=ventas<=20:
    salario=salario_base*0.2+salario_base
    print("el salario es -->",salario)
elif  21<=ventas<= 50:
    salario2=salario_base*0.3+salario_base
    print("el salario es:--->",salario2)
elif ventas> 51:
    salario3=salario_base*0.35+salario_base
    print("su salario es:",salario3)






