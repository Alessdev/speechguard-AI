import json
import pandas as pd
from pathlib import Path
from .speech_rules import SPEECH_SECTIONS
from .llm_utils import load_prompt, build_chain

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "speech_compliance_prompt.txt"


def build_speech_rules_text():
    """
    Convierte las reglas del speech en un texto estructurado
    que se enviará al LLM local.
    """
    text = []
    for sec in SPEECH_SECTIONS:
        text.append(f"- {sec['id']}: {sec['description']}")
    return "\n".join(text)


def evaluate_speech_compliance(grouped_df: pd.DataFrame):
    """
    Evalúa el cumplimiento del speech del asesor usando un modelo local (Ollama).
    Retorna un DataFrame con:
    - Cumplimiento global
    - Comentarios
    - Raw JSON de respuesta
    """
    prompt_template_str = load_prompt(PROMPT_PATH)
    chain = build_chain(prompt_template_str)

    results = []

    for _, row in grouped_df.iterrows():
        conv_id = row["conversation_id"]
        asesor_text = row["asesor_text"]

        rules_text = build_speech_rules_text()

        # Ejecutar el modelo local (phi3, mistral, llama3, etc.)
        response = chain.invoke({
            "speech_rules": rules_text,
            "asesor_text": asesor_text,
        })

        # Intentar convertir la salida en JSON
        try:
            data = json.loads(response)  # Con Ollama es un string plano
        except Exception as e:
            print(f"[WARN] Error parseando JSON en conv_id={conv_id}: {e}")
            print("Respuesta completa del modelo:")
            print(response)
            data = {
                "cumplimiento_global": "DESCONOCIDO",
                "comentario_global": "No se pudo interpretar el JSON devuelto.",
                "secciones": []
            }

        # Guardar resultados
        results.append({
            "conversation_id": conv_id,
            "speech_eval_raw": json.dumps(data, ensure_ascii=False),
            "cumplimiento_global": data.get("cumplimiento_global", "DESCONOCIDO"),
            "comentario_global": data.get("comentario_global", ""),
        })

    return pd.DataFrame(results)
