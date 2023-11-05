## Se sugiere la siguiente forma de organizar la data:
**Esta es la estructura que creará automáticamente el método principal en main**

Data
|   data1.csv
|   data2.csv
|   ...

Al binarizar con binarize.py se crea una carpeta por csv donde cada columna del csv tendrá un txt asignado
Bin: Data binarizada
|   data1.csv
|   |   1.txt
|   |   2.txt
|   |   ...
|   data2.csv
|   |   ...

LZ: Carpeta para guardar las cosas relacionadas a LZ
|   LZ76: Data procesada que devuelve LZ
|   |   data1.csv
|   |   |   1.lz76
|   |   |   2.lz76
|   |   ...
|   json: Data de LZ en forma de json 
|   |   data1.csv
|   |   |   1.json
|   |   |   2.json
|   |   ...
|   plots:
|   |   dimension_1
|   |   ...
|   |   dimension_1.png
|   |   ...


