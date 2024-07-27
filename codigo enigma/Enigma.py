def descifrar_mensaje(cadena_cifrada):
    # Alfabeto 
    alfabeto = 'abcdefghijklmnopqrstuvwxyz'
    
    # Función para obtener del alfabeto la posicion de la letra
    def indice_letra(letra):
        # Devuelve la posición de la letra en la cadena alfabeto
        return alfabeto.index(letra)
    
    # Función para descifrar una letra basada en el desplazamiento y dirección
    def descifrar_letra(letra, desplazamiento, direccion):
        # Obtiene el índice actual de la letra
        idx = indice_letra(letra)
        
        # Calcula el nuevo índice basado en la dirección del desplazamiento
        if direccion == 'adelante':
            nuevo_idx = (idx - desplazamiento) % 26
        else:  # Si la dirección es 'atras'
            nuevo_idx = (idx + desplazamiento) % 26
        
        # Devuelve la letra correspondiente al nuevo índice
        return alfabeto[nuevo_idx]
    
    # Lista para almacenar el mensaje descifrado
    mensaje_descifrado = []
    
    # Itera sobre cada letra en la cadena cifrada junto con su índice
    for i, letra in enumerate(cadena_cifrada):
        if letra.isalpha():  # Solo descifra letras
            # El desplazamiento es igual al índice + 1
            desplazamiento = i + 1
            
            # Alterna entre 'adelante' y 'atras' basado en el índice 
            direccion = 'adelante' if i % 2 == 1 else 'atras'
            
            # Descifra la letra usando el desplazamiento y dirección
            letra_descifrada = descifrar_letra(letra, desplazamiento, direccion)
            
            # Agrega la letra descifrada a la lista
            mensaje_descifrado.append(letra_descifrada)
        else:
            # Agrega caracteres no alfabéticos sin cambios
            mensaje_descifrado.append(letra)
    
    # Une la lista de caracteres en una cadena y la devuelve
    return ''.join(mensaje_descifrado)

# Ejemplo de uso
mensaje_cifrado = "ggipj_7"  # Mensaje cifrado 
mensaje_descifrado = descifrar_mensaje(mensaje_cifrado)
print(mensaje_descifrado)  
