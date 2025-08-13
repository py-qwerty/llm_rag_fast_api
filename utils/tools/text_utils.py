from typing import List


def smart_chunk_text(text: str, max_words: int = 1000) -> List[str]:
    """
    Divide un texto en chunks de hasta max_words palabras.
    Trata de respetar párrafos y oraciones para mantener la coherencia.
    """

    print(text)
    # 1. Separar en párrafos
    # paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # chunks = []
    # current_chunk = []
    # current_words = 0

    # for paragraph in paragraphs:
    #     # Contar palabras del párrafo
    #     paragraph_words = len(paragraph.split())

    #     # Si agregar el párrafo supera el límite, cerrar el chunk actual
    #     if current_words + paragraph_words > max_words:
    #         if current_chunk:
    #             chunks.append(" ".join(current_chunk))
    #         # Si el párrafo es demasiado grande, dividirlo por oraciones
    #         if paragraph_words > max_words:
    #             sentences = re.split(r'(?<=[.!?]) +', paragraph)
    #             temp_chunk = []
    #             temp_words = 0
    #             for sentence in sentences:
    #                 sentence_words = len(sentence.split())
    #                 if temp_words + sentence_words > max_words:
    #                     if temp_chunk:
    #                         chunks.append(" ".join(temp_chunk))
    #                     temp_chunk = [sentence]
    #                     temp_words = sentence_words
    #                 else:
    #                     temp_chunk.append(sentence)
    #                     temp_words += sentence_words
    #             if temp_chunk:
    #                 chunks.append(" ".join(temp_chunk))
    #             current_chunk = []
    #             current_words = 0
    #         else:
    #             current_chunk = [paragraph]
    #             current_words = paragraph_words
    #     else:
    #         current_chunk.append(paragraph)
    #         current_words += paragraph_words

    # # Agregar el último chunk
    # if current_chunk:
    #     chunks.append(" ".join(current_chunk))

    # return chunks