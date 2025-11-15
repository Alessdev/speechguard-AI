import json
import pandas as pd
from pathlib import Path
from .llm_utils import build_chain, load_prompt

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "conversation_analysis_prompt.txt"


def analyze_conversations_semantics(grouped_df: pd.DataFrame):
    """
    Analiza la conversación completa (cliente + asesor) y extrae información:
    - Resumen
    - Intención principal del cliente
    - Sentimiento
    - Temas principales
    - Frases clave
    - Oportunidades de mejora del asesor

    Compatible con modelos locales vía Ollama.
    """
    prompt_template_str = load_prompt(PROMPT_PATH)
    chain = build_chain(prompt_template_str)

    results = []

    for _, row in grouped_df.iterrows():
        conv_id = row["conversation_id"]
        full_conv = row["full_conversation"]

        # Ejecutar el modelo local
        response = chain.invoke({"full_conversation": full_conv})

        # Intentar interpretar la respuesta como JSON
        try:
            data = json.loads(response)  # Ollama devuelve texto directo
        except Exception as e:
            print(f"[WARN] No se pudo parsear JSON en conv_id={conv_id}: {e}")
            print("Respuesta completa del modelo:")
            print(response)
            data = {
                "resumen": "",
                "intencion_principal_cliente": "",
                "sentimiento_cliente": "DESCONOCIDO",
                "temas_principales": [],
                "frases_clave_cliente": [],
                "oportunidades_mejora_asesor": []
            }

        # Convertimos listas a strings para guardar en CSV
        results.append({
            "conversation_id": conv_id,
            "semantics_raw": json.dumps(data, ensure_ascii=False),
            "resumen": data.get("resumen", ""),
            "intencion_principal_cliente": data.get("intencion_principal_cliente", ""),
            "sentimiento_cliente": data.get("sentimiento_cliente", ""),
            "temas_principales": ", ".join(data.get("temas_principales", [])),
            "frases_clave_cliente": " | ".join(data.get("frases_clave_cliente", [])),
            "oportunidades_mejora_asesor": " | ".join(data.get("oportunidades_mejora_asesor", [])),
        })

    return pd.DataFrame(results)
