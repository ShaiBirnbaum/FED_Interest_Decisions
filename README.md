**Project README**

**1. Project Objetive**
POC for Predicting Federal Interest Decisions Based on Federal Reserve Speeches:

    1.1 Predict Interest Decisions from Comprehensive Speech Content
    1.2 Real-Time Prediction of Interest Decisions as Speech Progresses
    1.3 Real-time Identification of Critical Interest Prediction Highlights

**2. Database**

    2.1 FED Speeches and Speakers (1996-2020)
    2.2 FED Interest Decisions (1996-2020)

**3. Development Panel**
All speeches delivered in the month preceding an interest decision were tagged with the corresponding decision.

**4. Dependent Variable**
Interest decisions were categorized into three classes: Increase, Decrease, and No Change.

**5. Modeling**
Based on the “Framing Evidence” protocol, semantic resemblance was measured between interest change "framing" sentences and the speech sentences:

    5.1 ChatGPT was used to generate 20 sentences (“Anchors”) likely related to:
        5.1.1 Interest increase (explicit changes)
        5.1.2 Interest decrease (explicit changes)
        5.1.3 Interest increase (implicit changes)
        5.1.4 Interest decrease (implicit changes)
    5.2 Anchors generated in section 5.1 were transformed into embeddings using the “distilbert-base-nli-stsb-mean-tokens” model.
    5.3 Speeches were chunked into sentences and then transformed into embeddings with the same model.
    5.4 Cosine similarity was computed for every sentence in each speech against all anchor sentences, and the highest values for each        speech were saved for every anchor.
    5.5 Each anchor sentence was used as a feature, along with its corresponding saved cosine similarity values for each speech               calculated in section 5.4.
    5.6 A multiclass LGBM classification model was trained to predict FED interest decisions: Increase, Decrease, or No Change.

**6. Metrics**
Performance was assessed using:

Accuracy
Recall
Precision
F1-Score

These could be improved in future studies

**7. Usability and Explainability**
The model can be integrated as a component of a broader feature set that includes macroeconomic data to predict interest decisions. It can also be used in real-time as speeches progress to predict future interest decisions and provide live highlights (e.g., “Investments in various sectors are experiencing a noticeable uptick, indicating increased business confidence”). A simulation of using the model to predict future interest decisions as a FED speech progresses is presented.

**8. Limitations and Future Study Directions**

    8.1 The number of observations is limited by actual FED interest decisions. Future studies could expand the dataset using speeches        and decisions from other countries' central banks.
    8.2 Methodologies suitable for analyses involving small sample sizes may be employed to address this challenge.
    8.3 This model relies solely on speeches to predict interest decisions. Since these decisions are influenced by macroeconomic             conditions, it is recommended to integrate this model with macroeconomic data models to enhance performance.
    8.4 Research different embedding models and adjust them to be more sensitive to macroeconomic information.
    8.5 adjust hyper parameters like anchor length etc.

**9. Contributions**
Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests to improve the project.

**10. License**
This project is licensed under the MIT License. See the LICENSE file for details.

