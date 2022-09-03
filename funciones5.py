decimal=9
conversion_binario=bin(decimal)
conversion_octal=oct(decimal)
conversion_hex=hex(decimal)

print("=== decimal ===>",decimal)
print("bin,oct,hex===>",conversion_binario,conversion_hex,conversion_octal)

#¿como hacer lo contrario ?
#bin==> decimal
#oct==> decimal
#hex==> decimal

bin,oct,hex="1100110","146","66"

print("bin a decimal==>", int(bin,2))
print("oct a decimal==>",int(oct,8))
print("hex a decimal==>",int(hex,16))
