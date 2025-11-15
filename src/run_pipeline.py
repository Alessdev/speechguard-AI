import pandas as pd
from .config import DATA_RAW_PATH, RESULTS_PATH
from .preprocessing import preprocess_conversations
from .speech_compliance import evaluate_speech_compliance
from .semantic_analysis import analyze_conversations_semantics

def main():
    print("Cargando datos...")
    df = pd.read_csv(DATA_RAW_PATH)

    print("Preprocesando texto...")
    df_pre, grouped = preprocess_conversations(df)

    print("Evaluando cumplimiento del speech...")
    speech_df = evaluate_speech_compliance(grouped)

    print("Analizando semántica...")
    sem_df = analyze_conversations_semantics(grouped)

    print("Uniendo resultados...")
    final = (
        grouped
        .merge(speech_df, on="conversation_id")
        .merge(sem_df, on="conversation_id")
    )

    print(f"Guardando resultados en {RESULTS_PATH}")
    final.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    print("Pipeline completado correctamente.")

if __name__ == "__main__":
    main()
