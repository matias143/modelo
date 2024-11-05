import pandas as pd
import jsonlines

# Ruta al archivo Excel
file_path = "C:/Users/matia/OneDrive/Escritorio/ia/InformacionIAordenado2.xlsx"

# Cargar el archivo Excel
data = pd.read_excel(file_path)

# Lista para almacenar los pares pregunta-respuesta
preguntas_respuestas = []

# Procesar cada fila del archivo
for idx, row in data.iterrows():
    cesfam = row['CESFAM']
    
    # Generar preguntas y respuestas para cada campo
    preguntas = [
        # Información básica del CESFAM
        {
            "contexto": cesfam,
            "pregunta": f"¿Dónde se ubica el {cesfam}?",
            "respuesta": row['Dirección'] if pd.notna(row['Dirección']) else "Información no disponible"
        },
        # Horarios
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuál es el horario de atención general del {cesfam}?",
            "respuesta": row['Horario Atención'] if pd.notna(row['Horario Atención']) else "Información no disponible"
        },
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuál es el horario de la farmacia en el {cesfam}?",
            "respuesta": row['Horario Farmacia'] if pd.notna(row['Horario Farmacia']) else "Información no disponible"
        },
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuál es el horario de urgencias en el {cesfam}?",
            "respuesta": row['Horario Urgencias'] if pd.notna(row['Horario Urgencias']) else "Información no disponible"
        },
        # Contacto
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuál es el contacto y el teléfono del {cesfam}?",
            "respuesta": f"Email: {row['Contacto']}, Teléfono: {row['Teléfono']}" if pd.notna(row['Contacto']) and pd.notna(row['Teléfono']) else "Información no disponible"
        },
        # Servicios
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuáles son los servicios principales ofrecidos por el {cesfam}?",
            "respuesta": row['Servicios Principales'] if pd.notna(row['Servicios Principales']) else "Información no disponible"
        },
        # Sectores o poblaciones
        {
            "contexto": cesfam,
            "pregunta": f"¿Qué sectores o poblaciones atiende el {cesfam}?",
            "respuesta": row['Sectores o Poblaciones Atendidas'] if pd.notna(row['Sectores o Poblaciones Atendidas']) else "Información no disponible"
        },
        # Requisitos de inscripción
        {
            "contexto": cesfam,
            "pregunta": f"¿Cuáles son los requisitos para inscribirse en el {cesfam}?",
            "respuesta": row['Requisitos de Inscripción'] if pd.notna(row['Requisitos de Inscripción']) else "Información no disponible"
        },
        # Observaciones adicionales
        {
            "contexto": cesfam,
            "pregunta": f"¿Qué observaciones especiales tiene el {cesfam}?",
            "respuesta": row['Observaciones'] if pd.notna(row['Observaciones']) else "Información no disponible"
        }
    ]
    
    # Agregar las preguntas a la lista general
    preguntas_respuestas.extend(preguntas)

# Guardar el archivo en formato JSONL
with jsonlines.open("preguntas_respuestas_detalladas.jsonl", mode='w') as writer:
    writer.write_all(preguntas_respuestas)

print("Archivo 'preguntas_respuestas_detalladas.jsonl' creado con éxito.")
