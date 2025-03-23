#!/bin/bash

# Crear el entorno virtual
python3 -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip

# Instalar dependencias
pip install -r requirements.txt

echo "Entorno virtual creado y dependencias instaladas correctamente."
echo "Para activar el entorno virtual, ejecuta: source venv/bin/activate" 