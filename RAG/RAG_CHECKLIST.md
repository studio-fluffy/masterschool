# RAG System - Einfache Checkliste

## 1️⃣ Daten vorbereiten

- [ ] Datenquellen sammeln (Dokumente, PDFs, Webseiten, Datenbanken)
- [ ] Fehler und Duplikate entfernen
- [ ] Datenformat einheitlich machen (z.B. UTF-8 Encoding)
- [ ] Metadaten hinzufügen (Datum, Quelle, Kategorie, Version)
- [ ] Regelmäßig aktualisieren (mindestens monatlich)

## 2️⃣ Text in Teile zerlegen

- [ ] Lange Texte in kleinere Stücke aufteilen
- [ ] Stückgröße testen (Standard: **512-1024 Zeichen** oder **100-200 Wörter**)
- [ ] Überlappung zwischen Stücken einstellen (Standard: **50-100 Zeichen** Überlappung)
- [ ] Spezialzeichen und Formatierung bereinigen
- [ ] Für längere Dokumente: zuerst nach Kapitel/Absatz teilen, dann fine-tunen

## 3️⃣ Embeddings erstellen

- [ ] Embedding-Modell auswählen (Standard: **Sentence-Transformers, OpenAI text-embedding-3-small, oder local: all-MiniLM-L6-v2**)
- [ ] Modell mit Sprache und Daten kompatibel (für Deutsch: auf Deutsch trainierte Modelle)
- [ ] Kosten prüfen (Local = kostenlos, API ca. $0.01-0.10 pro 1M Tokens)
- [ ] Lokale oder Cloud-Lösung (Local für Datenschutz, Cloud für höhere Qualität)

## 4️⃣ Vektordatenbank einrichten

- [ ] Datenbank aussuchen (Standard: **Chroma für Anfänger, Pinecone für Produktion, Weaviate für große Systeme**)
- [ ] Datenbank mit Daten füllen (indexieren)
- [ ] Backup erstellen (automatisch täglich)
- [ ] Geschwindigkeit testen (Standard: <500ms Query-Zeit anstreben)

## 5️⃣ Suche konfigurieren

- [ ] Wie viele Ergebnisse zurückbringen? (Standard: **Top 5-10 Ergebnisse**)
- [ ] Wie relevant müssen Ergebnisse sein? (Standard: **Ähnlichkeit >0.7** akzeptieren)
- [ ] Qualität der Suchergebnisse prüfen (Test mit 10-20 Beispiel-Fragen)

## 6️⃣ AI-Modell verbinden

- [ ] KI-Modell aussuchen (Standard: **GPT-4o mini für Kosten, GPT-4o für Qualität, oder Ollama/Llama2 lokal**)
- [ ] Anweisung schreiben, was die KI machen soll (Prompt Engineering)
- [ ] Kosten berechnen (z.B. GPT-4o mini: ~$0.15 pro 1M Input Tokens)
- [ ] Parameter einstellen (Temperature: **0.3-0.7** für Balance, Max Tokens: **500-1000**)

## 7️⃣ Testen & Verbessern

- [ ] Test-Fragen vorbereiten (mindestens **20-50 diverse Fragen**)
- [ ] Antworten der KI überprüfen (Score: 1-5, ab 4 ist gut)
- [ ] Fehler dokumentieren (welche Fragen gehen schief?)
- [ ] Einstellungen anpassen (Chunking, Top-K, Prompts)
- [ ] Wieder testen und iterieren (bis Score >4.0)

## 8️⃣ Sicherheit

- [ ] Passwörter & API-Keys sicher speichern (in **.env Datei**, nicht in Code!)
- [ ] Wer darf das System benutzen? (Authentifizierung, z.B. Login)
- [ ] Datenschutz beachten (GDPR: Daten nicht unbegrenzt speichern)
- [ ] Rate Limiting (max. 100 Requests pro Minute pro User)

## 9️⃣ Produktiv nehmen

- [ ] System live machen (mit Monitoring!)
- [ ] Fehler überwachen (Error-Rate: sollte <5% sein)
- [ ] Logs speichern (alle Anfragen und Antworten tracken)
- [ ] Dokumentation schreiben (für andere Entwickler und Support)
- [ ] Response Time monitoren (Standard: <2 Sekunden)

## 🔟 Regelmäßig kontrollieren

- [ ] Wie gut funktioniert das System? (Standard: **User-Zufriedenheit >80%**, Error-Rate <5%)
- [ ] Fehler beheben (mindestens wöchentlich)
- [ ] Neue Daten hinzufügen (mindestens monatlich aktualisieren)
- [ ] Verbesserungen implementieren (A/B Tests mit neuen Parametern)
- [ ] Kosten überprüfen (Budget im Auge behalten)
