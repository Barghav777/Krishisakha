# KrishiSakha

KrishiSakha is conceptualized as a proactive, multilingual, and agentic artificial intelligence advisory system. Its design objective is to function as a sophisticated informational resource for agricultural practitioners in India, delivering timely, individualized, and empirically grounded guidance to augment agronomic practices. This initiative was conceived and executed in the context of the Capital One Launchpad Hackathon.

## Demo Link
[[https://huggingface.co/spaces/Barghav777/KrishiSakha]]

## Abstract of the Solution

The KrishiSakha system is engineered to address a series of profound challenges endemic to the Indian agricultural sector, including but not limited to climatic volatility, significant crop attrition, and informational asymmetries exacerbated by linguistic and digital disparities. The application is designed to accept multimodal data inputs—encompassing vocalizations, textual entries, and pictorial representations across a spectrum of more than twenty indigenous Indian languages—and to subsequently furnish actionable intelligence. This is achieved through the synthesis of real-time data streams from public-sector sources, such as meteorological forecasts and soil health repositories, with a validated corpus of agricultural science, thereby facilitating the generation of dependable responses and preemptive notifications.

In a departure from conventional question-and-answer paradigms, KrishiSakha operates as an agentic framework that curates stateful "Farmer Profiles." Such profiles enable the dissemination of timely alerts pertaining to prospective pestilential outbreaks, severe meteorological events, or advantageous market fluctuations, thereby empowering agriculturalists to execute decisions predicated on empirical data.

## Core System Capabilities

* **Multimodal and Multilingual Interfacing:** Interaction with the system is facilitated through vocal or textual inputs in the user's vernacular. The system incorporates advanced computational models to comprehend regional linguistic and dialectal variations.

* **Image-Based Pathological Analysis:** The submission of a botanical photographic specimen, specifically of a plant leaf, initiates an artificial intelligence-driven analysis to ascertain the presence of potential phytopathologies or pestilential agents, thereby furnishing immediate diagnostic assistance.

* **Hyperlocal Data Synthesis:** Advisory outputs are precisely calibrated to the user's specific geographical coordinates. The system effectuates the integration of real-time data sets, which include meteorological projections, localized soil condition assessments, and prevailing market valuations sourced from Agmarknet.

* **Proactive Agentic Framework:** The system is capable of issuing timely notifications, deliverable via Short Message Service (SMS) or in-application alerts, regarding potential agricultural risks and opportunities. It anticipates the informational requirements of the user based upon their profile and contemporaneous data, thereby transitioning from a decision-support to a decision-automation modality.

* **Factual Grounding and Verification of Responses:** Through the implementation of a Retrieval-Augmented Generation (RAG) architecture, system-generated responses are rigorously grounded in a corpus of authenticated agricultural research documents, sourced from institutions such as the Indian Council of Agricultural Research (ICAR), which serves to minimize the propagation of misinformation and confabulation.

* **Explainable Artificial Intelligence (XAI):** Each recommendation is accompanied by a mechanism for elucidation, colloquially termed a "Why?" feature, which articulates the specific data points that informed the resultant advisory. This functionality is intended to foster transparency and user confidence in the system.

## Architectural Framework

The system's architecture is predicated upon a **Core Intelligence Engine**, which serves to orchestrate the complex interactions among a plurality of data sources and specialized computational models.

1.  **Multimodal Data Ingress:** User interaction is initiated through one of several modalities: Text, Voice, Image, or SMS.

2.  **Core Intelligence Engine:**

    * **Linguistic Processing:** Input data undergoes a sequence of processing steps, including transcription and translation into a standardized analytical language (English).

    * **Intent Classification and Model Routing:** The system performs a classification of the user's intent (e.g., meteorological inquiry, pathological identification, cropping advice) and subsequently routes the query to the appropriate specialist model or data repository.

3.  **Data and Model Stratum:**

    * **Data Repositories:** The engine retrieves contextual information from a **Knowledge Base**, which is interrogated via the RAG mechanism, and from **Real-Time Data** Application Programming Interfaces (APIs) for weather, soil, and market data.

    * **Specialist Computational Models:** The query is dispatched to models specialized for tasks such as factual question-answering, plant science analysis, pestilence identification, or predictive forecasting.

4.  **Synthesis and Verification Layer:** The outputs from all constituent sources are synthesized into a coherent and unified response. This architectural layer performs a verification of the synthesized information against the established knowledge base to ensure a high degree of accuracy.

5.  **Response Generation and Delivery:** A comprehensive and actionable response is formulated in the user's original language and is subsequently delivered through the appropriate communication channel.

## Technical Specification

The platform has been constructed utilizing a foundation of robust, open-source technological components.

### Backend Infrastructure (`Python/Flask`)

* **Application Programming Interface Server:** A resilient API server, constructed with the **Flask** framework, is responsible for the management of all incoming network requests, the orchestration of the artificial intelligence pipeline, and the serving of the frontend user interface.

* **Input Processing Pipeline:** A sophisticated data processing pipeline, codified in `input_processing.py`, manages the multifaceted task of processing multimodal inputs. This includes, inter alia, audio signal enhancement utilizing the `pydub` library and the extraction of textual data.

### Artificial Intelligence and Machine Learning Models

* **Core Logic: Retrieval-Augmented Generation (RAG)**

    * **Vector Database:** A `FAISS` (Facebook AI Similarity Search) index, constructed from a knowledge base of agricultural research literature, is employed.

    * **Embedding Model:** The `BAAI/bge-large-en-v1.5` model is utilized for the generation of dense vector representations of text, enabling semantic search functionalities.

    * **Generator Model:** The `Groq API` provides access to the `llama-3.3-70b-versatile` large language model, which is leveraged to synthesize contextually rich and factually accurate responses.

* **Linguistic and Speech Processing**

    * **Automatic Speech Recognition (ASR):** The `OpenAI Whisper (small)` model is implemented for the transcription of audio inputs across various Indian languages.

    * **Language Identification (LID):** The `facebook/mms-lid-126` model is used for audio inputs, while the `langdetect` library is used for textual inputs.

    * **Translation and Natural Language Processing (NLP):** The `Groq API (Llama 3)` is employed for high-fidelity translation and advanced natural language understanding tasks.

    * **Optical Character Recognition (OCR):** The `EasyOCR` library is integrated for the extraction of textual information from user-submitted images.

* **Specialist Models**

    * **Phytopathological Detection:** A fine-tuned **EfficientNet-B0** convolutional neural network, trained on a dataset comprising 38 distinct classes of plant diseases, is utilized.

    * **Web Search Integration:** The `SerpApi` is employed to retrieve real-time market price data and other contemporaneous information that may not be present within the static knowledge base.

### Frontend Infrastructure (`HTML/CSS/JS`)

* **User Interface:** The system presents a clean, responsive, and mobile-centric user interface, featuring a primary chat area supplemented by sidebars for conversation history and frequently asked questions.

* **Dynamic Functionalities:** The frontend incorporates a theme switcher for dark and light modes, client-side voice recording capabilities through the `MediaRecorder` API, and text-to-speech output generation via the browser's native `SpeechSynthesis` API.

## Ethical Considerations and Mitigation Strategies

* **Risk: Artificial Intelligence Confabulation.**

    * **Mitigation:** All system-generated responses are strictly grounded in the verified knowledge base through the RAG architecture. A dedicated verification layer is included to maintain factual accuracy.

* **Risk: Data Privacy and Confidentiality.**

    * **Mitigation:** All user-associated data is subjected to anonymization procedures. Users are afforded complete control over their respective profiles and data.

* **Ethical Consideration: Inclusivity and Accessibility.**

    * **Mitigation:** A foundational principle of the system's design is its focus on multilingualism, a voice-first interaction modality, and compatibility with low-technology access channels such as SMS. This reflects a commitment to the equitable distribution of artificial intelligence benefits.

* **Ethical Consideration: Accountability.**

    * **Mitigation:** The system is explicitly positioned as a **decision-support instrument** and is not intended to supersede human judgment. The ultimate responsibility for all decisions remains with the agricultural practitioner.

## Contributing Personnel

* Himanshu

* Purushartha Gupta

* Barghav

* Nitin Kumar Yadav
