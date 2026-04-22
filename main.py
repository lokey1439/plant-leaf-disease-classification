import streamlit as st
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("model.weights.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("AGRO-FARM CARE – CROP, FERTILIZER & LEAF DISEASE PREDICTION")
    image_path = ("home_page.jpeg")
    st.image(image_path,use_column_width=True)
    st.markdown("""
    **Welcome to the Plant Disease Recognition System!** 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = [
            '***APPLE APPLE SCAB*** Apple scab is a common and serious fungal disease of apple trees caused by the fungus Venturia inaequalis. It thrives in cool, wet weather, especially during spring when new growth is emerging. The fungus overwinters on fallen leaves and infected twigs, and releases spores that infect young leaves, flowers, and fruits. Symptoms include olive-green to black spots on leaves and fruits, leading to leaf drop and deformed, cracked apples....'
                "###CAUSE OF APPLE SCAB DISEASE:"
                    "Caused by the fungus Venturia inaequalis."
                    "Spreads in cool, moist weather (especially spring)."
                    "Overwinters on fallen leaves and infected twigs."
                    "Symptoms: black/olive spots on leaves and fruit, leaf drop, fruit deformity."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Clean and destroy fallen leaves in autumn."
                    "Prune trees to improve air circulation."
                    "Use scab-resistant apple varieties (e.g., ‘Liberty’, ‘Freedom’)."
                    "Ensure proper spacing between trees."
                    "Apply biological control agents like Trichoderma."
                "###PREVENTION WITH FERTILIZERS:"
                    "Avoid excessive nitrogen application."
                    "Use balanced NPK fertilizers to promote plant health."
                    "Apply protective fungicides (e.g., Captan, Mancozeb)."
                    "Rotate fungicides to prevent resistance."
                    "Use organic sprays like sulfur or potassium bicarbonate.",
            '***Apple Black rot*** '
                "###CAUSE OF THE DISEASE:"
                    "Caused by the fungus Botryosphaeria obtusa Infects leaves, fruit, bark, and twigs."
                    "Overwinters in dead wood, mummified fruits, and cankers."
                    "Spreads via wind and rain, especially in warm, moist weather."
                    "Symptoms: “Frog-eye” leaf spots, sunken fruit lesions, bark cankers, and twig dieback."
                "###PREVENION WITHOUT FERTILIZERS:"
                    "Remove and destroy infected leaves, twigs, and fruits."
                    "Prune trees to improve air circulation and light penetration."
                    "Use resistant or tolerant apple varieties. Avoid injuries to bark and fruit."
                    "Practice good orchard sanitation regularly."
                "###PREVENTION WITH FERTILIZERS:"
                    "Avoid excessive nitrogen application."
                    "Apply balanced NPK fertilizers to support tree health."
                    "Combine fertilization with fungicides like copper or captan."
                    "Use an integrated disease management approach."
                    "Strengthen tree immunity through proper nutrition and care.", 
            '***APPLE CEDAR APPLE RUST***'
                "###CAUSE OF THE DISEASE:"
                    "Caused by the fungus Gymnosporangium juniperi-virginianae."
                    "Requires two hosts: apple and juniper (cedar) trees."
                    "Overwinters as galls on cedar trees."
                    "Orange, jelly-like spore horns appear on galls in spring."
                    "Spores spread to apples causing orange leaf spots and fruit deformity."
                "###PREVENION WITHOUT FERTILIZERS:"
                    "Remove nearby cedar/juniper trees (300–500 m away)."
                    "Prune and destroy infected branches and galls."
                    "Plant resistant apple varieties (e.g., ‘Liberty’, ‘Enterprise’)."
                    "Maintain good airflow by pruning apple trees."
                    "Regular orchard sanitation and inspection."
                "###PREVENTION WITH FERTILIZERS:"
                    "Apply balanced NPK fertilizers to boost tree immunity."
                    "Avoid excess nitrogen to prevent soft, susceptible growth."
                    "Use fungicides (e.g., myclobutanil, mancozeb) during spring."
                    "Integrate cultural, nutritional, and chemical controls."
                    "Strengthen overall tree health to reduce vulnerability.", 
            '***APPLE HEALTHY***'
                ,
            '***BLUEBERRY HEALTHY***'
                    ,
            '***CHERRY (INCLUDING SOUR) – POWDERY MILDEW***'
                    "Powdery mildew is a common fungal disease affecting cherry trees, especially sour varieties. It is caused by the fungus Podosphaera clandestina, which thrives in warm, dry climates with high humidity during the night. The disease primarily affects young shoots, leaves, and fruits, forming a white to gray powdery coating on the surface. Infected leaves may curl, twist, or drop prematurely. Fruits can become russeted, cracked, or stunted. While it may not kill the tree, powdery mildew significantly weakens the plant, reduces fruit quality, and impairs overall vigor, making the tree more susceptible to other stressors."
                "###CAUSE OF POWDERY MILDEW DISEASE:"
                    "Caused by the fungus Podosphaera clandestina."
                    "Favors warm days and humid nights (especially late spring to summer)."
                    "Spreads via airborne spores and survives on twigs or fallen leaves."
                    "Symptoms: white powdery patches on leaves, shoots, and young fruits; leaf curl and fruit deformation."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Prune trees to allow better air circulation and light penetration."
                    "Remove and destroy infected leaves and shoots."
                    "Avoid overhead watering to reduce humidity on leaf surfaces."
                    "Use resistant cherry cultivars if available."
                    "Practice orchard hygiene—clean up leaf litter and fallen fruit."
                "###PREVENTION WITH FERTILIZERS:"
                    "Apply balanced fertilizers to support healthy growth (avoid excess nitrogen)."
                    "Use fungicides such as sulfur, potassium bicarbonate, or horticultural oils early in the season."
                    "For severe cases, systemic fungicides like myclobutanil may be used as per local guidelines."
                    "Maintain optimal soil fertility based on a soil test to enhance plant resilience."
                    "Combine fertilization with timely sprays during humid conditions for best control.", 
            '***CHERRY (INCLUDING SOUR) – HEALTHY***'
                    ,
            '***CORN (MAIZE) – CERCOSPORA LEAF SPOT / GRAY LEAF SPOT***'
                    "Cercospora leaf spot, commonly referred to as Gray Leaf Spot (GLS), is a destructive fungal disease of corn caused by Cercospora zeae-maydis. It thrives in warm, humid conditions and is particularly aggressive in dense plantings or no-till fields where crop residue remains on the soil surface. The disease first appears as small, rectangular gray or tan lesions on lower leaves and progressively moves upward. Severe infections can lead to extensive leaf necrosis, reducing the plant’s ability to photosynthesize, which directly impacts grain yield. GLS is now one of the most economically significant foliar diseases of corn worldwide."
                "###CAUSE OF CERCOSPORA / GRAY LEAF SPOT DISEASE:"
                    "Caused by the fungus Cercospora zeae-maydis."
                    "Survives on infected corn residue left on the field."
                    "Favors warm (25–30°C), humid conditions and poor air circulation."
                    "Spread by windborne spores during wet or foggy weather."
                    "Symptoms: long, narrow, rectangular gray lesions on leaves; leaf drying or death."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Rotate crops—avoid continuous corn planting in the same field."
                    "Bury crop residues by plowing to reduce overwintering fungi."
                    "Plant disease-resistant corn hybrids."
                    "Space plants properly to improve airflow between rows."
                    "Use clean, certified seeds to prevent initial infection sources."
                    "Monitor fields regularly and remove infected plants if possible."
                "PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Use balanced NPK fertilizers like 20-20-20 to strengthen plant health."
                    "Avoid excessive nitrogen, which can increase susceptibility to GLS."
                    "Apply potassium-based fertilizers (e.g., MOP – Muriate of Potash) to improve leaf strength."
                    "Use micronutrient mixes (with zinc, manganese, copper) to support resistance."
                    "Combine with protective fungicides like Strobilurins (e.g., Azoxystrobin) or Triazoles (e.g., Propiconazole) available at agrochemical stores."
                    "Begin fungicide applications at VT (tasseling) stage if disease pressure is high."
                    "Always read and follow label instructions for fungicide application timing and rotation.", 
            '***CORN (MAIZE) – COMMON RUST***'
                    "Common rust is a widespread foliar disease of corn caused by the fungus Puccinia sorghi. It affects both sweet and field corn, appearing most often in cooler, moist environments. The disease is characterized by the presence of small, reddish-brown pustules (spores) that appear on both the upper and lower surfaces of the leaves. As the infection progresses, the pustules darken and may merge, leading to premature leaf death and reduced photosynthesis. Although not as destructive as other corn diseases like southern rust or gray leaf spot, severe infections can still reduce grain yield and crop quality, especially in susceptible hybrids."
                "###CAUSE OF COMMON RUST DISEASE:"
                    "Caused by the fungus Puccinia sorghi."
                    "Spreads through airborne spores from southern regions during early growth."
                    "Favors cool temperatures (16–23°C) with high humidity or frequent dew."
                    "Spores germinate and infect leaves under wet conditions."
                    "Symptoms: scattered, oval reddish-brown pustules on leaves, later turning black."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Grow rust-resistant corn hybrids where available."
                    "Practice crop rotation to reduce disease pressure."
                    "Remove infected plant residues after harvest."
                    "Avoid planting too early in regions with a history of rust."
                    "Ensure good plant spacing for better airflow and faster leaf drying."
                    "Monitor early growth stages for signs of infection."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Apply balanced fertilizers like NPK 19-19-19 to promote healthy foliage."
                    "Supplement with potassium-based fertilizers such as SOP (Sulfate of Potash) to boost disease resistance."
                    "Avoid over-applying nitrogen (e.g., urea) which can promote soft tissue prone to infection."
                    "Use fungicides like Triazoles (e.g., Propiconazole) or Strobilurins (e.g., Azoxystrobin) available in most agro shops for foliar protection."
                    "Begin fungicide application early at the V6 to V8 stage if rust appears or weather conditions are favorable."
                    "Combine fertilization with proper irrigation management to reduce leaf wetness duration.",
            '***CORN (MAIZE) – NORTHERN LEAF BLIGHT***'
                    "Northern Leaf Blight (NLB) is a major foliar disease of corn caused by the fungus Exserohilum turcicum. It commonly affects corn in cool, wet climates and is particularly severe in regions with prolonged leaf wetness from rain or dew. The disease is identified by large, cigar-shaped, gray-green to tan lesions that run parallel to the leaf veins. As lesions grow and coalesce, they can kill large portions of the leaf surface, reducing photosynthesis and grain fill. NLB can significantly reduce yields, especially if infection occurs before or during tasseling. High humidity, dense planting, and leftover crop residue encourage the spread of this disease."
                "###CAUSE OF NORTHERN LEAF BLIGHT DISEASE:"
                    "Caused by the fungus Exserohilum turcicum."
                    "Overwinters in infected corn residue in the field."
                    "Spreads by windblown spores during cool (18–27°C), moist weather."
                    "Symptoms: long, elliptical gray or tan lesions on lower leaves spreading upwards."
                    "Severe infections reduce photosynthetic activity and yield."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Rotate with non-host crops (e.g., soybeans) to reduce fungal carryover."
                    "Practice deep plowing to bury infected residues."
                    "Select and plant resistant or tolerant corn hybrids."
                    "Ensure proper spacing and row orientation for good air circulation."
                    "Scout fields regularly, especially in high-moisture environments."
                    "Delay irrigation or water early in the day to reduce prolonged leaf wetness."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Apply balanced fertilizers such as NPK 20-20-20 or NPK 19-19-19 to promote robust growth."
                    "Avoid excessive nitrogen application (e.g., urea) that can result in overly lush foliage, which is more susceptible."
                    "Use potassium-enriched fertilizers (e.g., MOP – Muriate of Potash) to improve stress tolerance."
                    "Apply fungicides like Strobilurins (e.g., Azoxystrobin) or Triazoles (e.g., Tebuconazole) available in agricultural shops."
                    "Begin fungicide treatments at early signs of disease, especially around the VT (tasseling) stage."
                    "Combine fungicide use with foliar micronutrient sprays (e.g., zinc, manganese) to support defense mechanisms.", 
            '***CORN (MAIZE) – HEALTHY***'
                    ,
            '***GRAPE – BLACK ROT***'
                    "Black rot is a common and damaging fungal disease of grapevines, caused by the pathogen Guignardia bidwellii. It thrives in warm, humid conditions and primarily affects the leaves, shoots, tendrils, and fruit. The disease starts as small brown spots on leaves that develop dark borders. On fruit, the infection causes sunken, black lesions that eventually shrivel into hard, black, raisin-like mummies. These mummified berries are a major source of spores for future infections. If not managed, black rot can destroy a large portion of the grape crop and weaken vines over time."
                "###CAUSE OF BLACK ROT DISEASE:"
                    "Caused by the fungus Guignardia bidwellii."
                    "Overwinters in infected mummified berries and canes."
                    "Spreads by rain-splashed spores during warm, wet weather (21–29°C)."
                    "Symptoms: brown spots with dark margins on leaves, black sunken spots on berries that shrivel and harden."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Remove and destroy mummified fruits and pruned diseased canes."
                    "Prune vines to increase sunlight exposure and airflow."
                    "Use resistant grape varieties where available."
                    "Apply organic fungicidal sprays like neem oil or copper-based products."
                    "Train vines properly to keep fruit off the ground."
                    "Avoid overhead irrigation that keeps foliage wet."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Use balanced NPK fertilizers like NPK 19-19-19 to maintain plant vigor."
                    "Apply MOP (Muriate of Potash) to enhance disease resistance and fruit quality."
                    "Supplement with micronutrients like calcium, boron, and zinc to strengthen tissue integrity."
                    "Use fungicides such as Mancozeb, Myclobutanil, or Tebuconazole, especially at pre-bloom and post-bloom stages."
                    "Combine with bio-fertilizers like Trichoderma or Pseudomonas fluorescens for natural fungal control."
                    "Maintain a fertilization schedule based on soil testing for optimal nutrition and disease resistance.",
            '***GRAPE – ESCA (BLACK MEASLES)***'
                    "Esca, also known as Black Measles, is a chronic and complex grapevine trunk disease caused by a combination of fungal pathogens, mainly Phaeomoniella chlamydospora, Phaeoacremonium aleophilum, and Fomitiporia mediterranea. It commonly affects mature grapevines, especially in hot, dry climates. The disease slowly damages the wood and vascular system of the vine, leading to leaf scorch, reduced vigor, and fruit spots. Affected leaves develop yellow or reddish interveinal areas bordered by green veins — a symptom called “tiger stripe.” In advanced cases, berries shrivel and die, and white rot may be seen inside the wood. The disease is difficult to eradicate and often leads to the gradual decline or death of vines."
                "###CAUSE OF ESCA (BLACK MEASLES) DISEASE:"
                    "Caused by a fungal complex:"
                    "Phaeomoniella chlamydospora."
                    "Phaeoacremonium aleophilum."
                    "Fomitiporia mediterranea."
                    "Enters through pruning wounds, especially in old vines."
                    "More common in dry, warm climates."
                    "Symptoms: “Tiger stripe” pattern on leaves. Dark streaks or white rot in the trunk. Black spots on berries and shriveling. Sudden vine collapse in severe cases."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Avoid pruning during wet weather and disinfect tools between cuts."
                    "Remove and destroy infected vines and wood."
                    "Apply pruning wound protectants or sealants."
                    "Train new shoots and replace severely affected vines early."
                    "Improve airflow by pruning and maintaining vine spacing."
                    "Monitor regularly for early symptoms (leaf streaks, berry spotting)."
                    "Use tolerant rootstocks and certified disease-free planting material."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Apply balanced fertilizers such as NPK 19-19-19 or NPK 20-20-20 to maintain overall vine health."
                    "Supplement with Calcium Nitrate to strengthen cell walls and vascular tissue."
                    "Use Potassium fertilizers (e.g., MOP or SOP) to enhance stress resistance."
                    "Foliar spray with micronutrients like zinc, boron, and magnesium for photosynthetic support."
                    "Apply bio-fungicides such as Trichoderma harzianum to pruning wounds to prevent fungal colonization."
                    "Use systemic fungicides like Tebuconazole (as a protectant, not curative) after pruning, where allowed."
                    "Perform soil testing to ensure proper nutrient balance and avoid over- or under-application.", 
            '***GRAPE – LEAF BLIGHT (ISARIOPSIS LEAF SPOT)***'
                    "Leaf blight in grapevines, commonly known as Isariopsis Leaf Spot, is caused by the fungus Isariopsis clavispora. It primarily affects grape leaves, particularly in humid, wet conditions, and can lead to premature defoliation. The disease appears as angular, brown to black spots on older leaves, often starting near the base of the plant. As the disease progresses, the affected leaves may turn yellow and drop, weakening the vine and reducing fruit quality. If left unmanaged, the disease can contribute to poor fruit ripening, lower yields, and increased vulnerability to other diseases and pests."
                "###CAUSE OF LEAF BLIGHT (ISARIOPSIS LEAF SPOT) DISEASE:"
                    "Caused by the fungus Isariopsis clavispora."
                    "Thrives in warm, moist environments, especially during frequent rainfall."
                    "Spreads via water splash and wind-borne spores."
                    "Symptoms: Angular brown to black spots on leaves. Yellowing and early leaf drop. Reduced plant vigor and lower fruit yield."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Regular pruning to improve airflow and reduce humidity around vines."
                    "Remove and destroy infected leaves and pruning debris."
                    "Space vines properly to allow good air circulation and sunlight penetration."
                    "Use resistant grape varieties if available."
                    "Avoid overhead irrigation, especially in late afternoon."
                    "Apply organic fungicides like copper-based sprays during early disease signs."
                    "Monitor the vineyard regularly during rainy seasons."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Use balanced NPK fertilizers (e.g., NPK 19-19-19, NPK 20-20-20) to strengthen vine growth."
                    "Apply MOP (Muriate of Potash) or SOP (Sulfate of Potash) to improve disease resistance and leaf strength."
                    "Include Calcium Nitrate to maintain healthy leaf tissues."
                    "Use foliar sprays containing zinc, manganese, and boron to support photosynthesis and immunity."
                    "Combine with bio-fungicides such as Trichoderma viride or Bacillus subtilis as a preventive approach."
                    "Apply fungicides such as Chlorothalonil, Mancozeb, or Azoxystrobin during pre- and post-rainfall periods."
                    "Follow soil and tissue testing guidelines to tailor fertilizer application accurately.", 
            '***GRAPE – HEALTHY***'
                    , 
            '***ORANGE – HUANGLONGBING (CITRUS GREENING)***'
                     "Huanglongbing (HLB), also known as Citrus Greening, is one of the most devastating citrus diseases worldwide. It is caused by the bacterium Candidatus Liberibacter asiaticus and is primarily spread by the Asian citrus psyllid (Diaphorina citri). The disease affects all citrus varieties and leads to severe fruit deformation, bitter taste, leaf yellowing, twig dieback, and eventual tree death. Once a tree is infected, there is no cure. Early detection and control of the psyllid vector are key to disease management."
                "###CAUSE OF HUANGLONGBING (CITRUS GREENING) DISEASE:"
                    "Caused by the phloem-limited bacterium Candidatus Liberibacter asiaticus."
                    "Spread by the Asian citrus psyllid (Diaphorina citri)."
                    "Symptoms: Mottled yellow leaves resembling nutrient deficiency. Small, misshapen, bitter-tasting fruits with green color remaining at the stem end greening. Poor flowering and fruit set. Twig dieback and general tree decline. Once infected, trees cannot be cured."
                "###PREVENTION WITHOUT FERTILIZERS:"
                    "Monitor for and control the Asian citrus psyllid using sticky traps and pruning."
                    "Remove and destroy infected trees promptly to prevent spread."
                    "Plant certified disease-free nursery stock."
                    "Use barrier crops or screen netting in nurseries and young orchards."
                    "Encourage natural predators like lady beetles to control psyllids."
                    "Practice good orchard sanitation by removing fallen leaves and twigs."
                    "Educate workers to identify early signs of infection."
                "###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
                    "Apply balanced NPK fertilizers (e.g., NPK 19-19-19 or NPK 8-24-24) to support overall tree health."
                    "Use Magnesium sulfate and Zinc sulfate foliar sprays to correct greening-like deficiency symptoms."
                    "Apply Micronutrient mixes with iron, manganese, and boron to support leaf health."
                    "Use Calcium Nitrate to strengthen tissues and improve nutrient transport."
                    "Apply Silicon-based fertilizers (e.g., potassium silicate) to boost disease resistance."
                    "Incorporate organic bio-fertilizers like Azospirillum and Phosphobacteria to improve root strength."
                    "Fertilizer support helps delay decline in infected trees but is not a cure — it only improves longevity and yield in early infection stages.",
        '***Peach___Bacterial_spot***'
                    "Bacterial spot is a common disease of peach caused by the bacterium Xanthomonas campestris pv. pruni. It mainly affects leaves, fruits, and twigs, leading to spots, fruit damage, and reduced yield. The disease spreads through rain, wind, and infected tools, especially in warm and wet conditions."
                "###CAUSE OF BACTERIAL SPOT DISEASE:"
"Caused by Xanthomonas campestris pv. pruni."
"Spreads through rain splash, wind, and contaminated tools."
"Symptoms include dark leaf spots, yellow halos, fruit cracks, and leaf drop."
"Common in warm and humid weather."
"###PREVENTION WITHOUT FERTILIZERS:"
"Plant resistant varieties."
"Prune for better air flow."
"Avoid overhead irrigation."
"Remove infected leaves and fruits."
"Disinfect tools regularly."
"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"
"Use balanced NPK fertilizers (19-19-19)."
"Apply Calcium nitrate to strengthen tissues."
"Spray Zinc, Boron, and Magnesium micronutrients."
"Use copper-based sprays for disease control."
"Bio-fertilizers improve plant health.",
                    
            '***Peach___healthy', 
            'Pepper,_bell___Bacterial_spot'
            "Bacterial spot is a common disease of bell pepper caused by the bacterium Xanthomonas campestris pv. vesicatoria. It affects leaves, stems, and fruits, causing spots, leaf drop, and reduced yield. The disease spreads quickly in warm, wet conditions through rain splash, wind, and infected seeds."
"###CAUSE OF BACTERIAL SPOT DISEASE:"
"Caused by Xanthomonas campestris pv. vesicatoria."
"Spreads through infected seeds, rain, wind, and tools."
"Symptoms include small dark leaf spots, yellowing, fruit lesions, and leaf drop."
"Common in warm and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Use disease-free seeds."
"Avoid overhead watering."
"Remove infected plants and leaves."
"Rotate crops regularly."
"Keep good spacing for airflow."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers (19-19-19)."
"Use Calcium nitrate to strengthen plant cells."
"Spray Zinc, Magnesium, and Boron micronutrients."
"Copper-based sprays help control bacteria."
"Bio-fertilizers improve plant resistance.", 

'Pepper,_bell___healthy', 
                    
            'Potato___Early_blight'
            "Early blight is a common fungal disease of potato caused by Alternaria solani. It mainly affects leaves, stems, and tubers, leading to dark spots, yellowing, and reduced yield. It spreads in warm, humid conditions and can weaken the plant over time."

"###CAUSE OF EARLY BLIGHT DISEASE:"

"Caused by fungus Alternaria solani."
"Spreads through wind, rain, and infected plant debris."
"Symptoms include brown leaf spots with rings, yellowing, and leaf drop."
"Tubers may develop dark, sunken lesions."
"Common in warm and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves and plants."
"Practice crop rotation."
"Avoid overhead irrigation."
"Maintain good plant spacing."
"Keep field clean from plant debris."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Use balanced NPK fertilizers (19-19-19)."
"Apply Potassium to improve plant strength."
"Use Calcium nitrate for tissue strength."
"Spray Magnesium and micronutrients (Zn, B)."
"Apply fungicides for better control.", 
            'Potato___Late_blight'
            "Late blight is a severe disease of potato caused by the fungus-like organism Phytophthora infestans. It mainly affects leaves, stems, and tubers, causing rapid decay and major yield loss. It spreads quickly in cool, wet weather and can destroy crops within days."

"###CAUSE OF LATE BLIGHT DISEASE:"

"Caused by Phytophthora infestans."
"Spreads through wind, rain, and infected plant material."
"Symptoms include dark water-soaked leaf spots, white fungal growth, and rotting tubers."
"Plants rapidly turn black and die."
"Common in cool and humid conditions."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected plants immediately."
"Avoid overhead irrigation."
"Use crop rotation."
"Maintain good spacing for airflow."
"Destroy plant debris after harvest."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers (19-19-19)."
"Use Potassium and Calcium nitrate for plant strength."
"Spray Copper-based fungicides for control."
"Use Magnesium and micronutrients (Zn, B)."
"Bio-fertilizers improve plant resistance.", 
            'Potato___healthy', 
            
            'Raspberry___healthy', 
            
            'Soybean___healthy', 
            
            '***Squash___Powdery_mildew***'
            "Powdery mildew is a common fungal disease of squash caused by fungi like Podosphaera xanthii. It affects leaves and stems, forming white powder-like growth on the plant surface. It reduces photosynthesis, weakens the plant, and lowers yield. It spreads easily in warm, dry conditions with high humidity."

"###CAUSE OF POWDERY MILDEW DISEASE:"

"Caused by fungi such as Podosphaera xanthii."
"Spreads through wind and infected plant debris."
"Symptoms include white powdery patches on leaves and stems."
"Leaves turn yellow, dry, and fall early."
"Common in warm weather with high humidity."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves."
"Ensure proper spacing for airflow."
"Avoid overhead irrigation."
"Keep field clean from weeds and debris."
"Grow resistant varieties."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Use balanced NPK fertilizers (19-19-19)."
"Apply Potassium to improve plant strength."
"Spray Sulfur-based fungicides for control."
"Use micronutrients like Zinc and Magnesium."
"Bio-fertilizers improve plant health and resistance.", 
                    '***Strawberry___Leaf_scorch***'
                    "Leaf scorch is a fungal disease of strawberry caused by Diplocarpon earlianum. It mainly affects leaves, causing dark purple or brown spots, leaf drying, and reduced plant growth. The disease spreads in warm, wet conditions and reduces fruit yield and quality."

"###CAUSE OF LEAF SCORCH DISEASE:"

"Caused by fungus Diplocarpon earlianum."
"Spreads through rain splash, wind, and infected debris."
"Symptoms include small purple spots on leaves, leaf browning, and drying."
"Severe infection leads to leaf drop and weak plants."
"Common in warm and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves."
"Avoid overhead irrigation."
"Maintain good spacing between plants."
"Keep field clean from debris."
"Use disease-free planting material."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers (19-19-19)."
"Use Potassium to improve plant strength."
"Spray Copper or Sulfur-based fungicides."
"Apply micronutrients like Zinc and Magnesium."
"Bio-fertilizers improve plant resistance.", 
            'Strawberry___healthy', 
            
            '***Tomato___Bacterial_spot***'
"Common in warm and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Use disease-free seeds."
"Avoid overhead irrigation."
"Remove infected plants and leaves."
"Rotate crops regularly."
"Keep proper spacing for airflow."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers (19-19-19)."
"Use Calcium nitrate to strengthen plant tissues."
"Spray Zinc, Boron, and Magnesium micronutrients."
"Copper-based sprays help control bacteria."
"Bio-fertilizers improve plant resistance.", 
                    '***Tomato___Early_blight***'
                    "Early blight is a fungal disease of tomato caused by Alternaria solani. It affects leaves, stems, and fruits, causing dark spots, yellowing, and reduced yield. It spreads in warm, humid conditions and can weaken plants over time."

"###CAUSE OF EARLY BLIGHT DISEASE:"

"Caused by fungus Alternaria solani."
"Spreads through wind, rain, and infected plant debris."
"Symptoms include brown leaf spots with concentric rings, yellowing, and leaf drop."
"Fruits may develop dark sunken spots."
"Common in warm and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves and plants."
"Practice crop rotation."
"Avoid overhead irrigation."
"Maintain good spacing for airflow."
"Keep field clean from debris."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Use balanced NPK fertilizers (19-19-19)."
"Apply Potassium for plant strength."
"Use Calcium nitrate for tissue health."
"Spray Magnesium and micronutrients (Zn, B)."
"Apply fungicides for better control.", 
            '***Tomato___Late_blight***'
            "Late blight is a very destructive disease of tomato caused by Phytophthora infestans. It affects leaves, stems, and fruits, causing rapid decay and heavy yield loss. It spreads quickly in cool, wet conditions and can destroy plants within a short time."

"###CAUSE OF LATE BLIGHT DISEASE:"

"Caused by Phytophthora infestans."
"Spreads through wind, rain, and infected plant material."
"Symptoms include dark water-soaked leaf spots, white fungal growth, and rotting fruits."
"Plants quickly turn black and die."
"Common in cool and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected plants immediately."
"Avoid overhead irrigation."
"Practice crop rotation."
"Maintain good spacing for airflow."
"Destroy plant debris after harvest."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers (19-19-19)."
"Use Calcium and Potassium for plant strength."
"Spray Copper-based fungicides for control."
"Apply Magnesium and micronutrients (Zn, B)."
"Bio-fertilizers improve plant resistance.", 
            '***Tomato___Leaf_Mold***'
            "Leaf mold is a fungal disease of tomato caused by Passalora fulva. It mainly affects leaves, especially in greenhouse or humid conditions. It reduces photosynthesis, weakens the plant, and lowers yield."

"###CAUSE OF LEAF MOLD DISEASE:"

"Caused by fungus Passalora fulva."
"Spreads through wind, water, and infected plant debris."
"Symptoms include yellow spots on upper leaf surface and grayish mold on the lower side."
"Leaves dry and fall early."
"Common in warm, humid, and poorly ventilated areas."

"###PREVENTION WITHOUT FERTILIZERS:"

"Improve air circulation between plants."
"Avoid overhead irrigation."
"Remove infected leaves."
"Keep field clean and dry."
"Use resistant varieties."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium for plant strength."
"Spray Copper or Sulfur-based fungicides."
"Apply micronutrients like Zinc and Magnesium."
"Bio-fertilizers improve plant health and resistance.", 
                    '***Tomato___Septoria_leaf_spot***'
                    "Septoria leaf spot is a fungal disease of tomato caused by Septoria lycopersici. It mainly affects leaves, causing small dark spots and heavy leaf drop. It reduces plant strength and fruit yield."

"###CAUSE OF SEPTORIA LEAF SPOT DISEASE:"

"Caused by fungus Septoria lycopersici."
"Spreads through rain splash, wind, and infected debris."
"Symptoms include small round spots with gray centers and dark edges."
"Leaves turn yellow and fall early."
"Common in warm, wet conditions."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves."
"Avoid overhead irrigation."
"Practice crop rotation."
"Keep good spacing for airflow."
"Clean plant debris regularly."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium for plant strength."
"Spray Copper-based fungicides."
"Apply Zinc and Magnesium micronutrients."
"Bio-fertilizers improve plant resistance.", 
            '***Tomato___Spider_mites Two-spotted_spider_mite***'
            "Spider mites are tiny pests (Tetranychus urticae) that attack tomato plants by feeding on leaf sap. They cause leaf damage, weakening the plant and reducing yield. Infestations are common in hot, dry conditions."

"###CAUSE OF SPIDER MITE INFESTATION:"

"Caused by two-spotted spider mite (Tetranychus urticae)."
"Spreads through wind, infested plants, and farm tools."
"Symptoms include tiny yellow or white spots on leaves, fine webbing, and leaf drying."
"Severe attack leads to leaf drop and weak plants."
"Common in hot and dry weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove heavily infested leaves."
"Spray water to reduce mite population."
"Keep plants well-irrigated (avoid dryness)."
"Maintain field cleanliness."
"Encourage natural predators like lady beetles."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium to improve plant strength."
"Apply Sulfur-based sprays for mite control."
"Use micronutrients like Zinc and Magnesium."
"Bio-fertilizers help improve plant resistance.", 
                    '***Tomato___Target_Spot***'
                    "Target spot is a fungal disease of tomato caused by Corynespora cassiicola. It affects leaves, stems, and fruits, causing dark spots and leaf drop. It reduces plant health and yield, especially in warm and humid conditions."

"###CAUSE OF TARGET SPOT DISEASE:"

"Caused by fungus Corynespora cassiicola."
"Spreads through wind, rain, and infected plant debris."
"Symptoms include round brown spots with concentric rings on leaves."
"Leaves turn yellow and fall early."
"Common in warm, wet, and humid weather."

"###PREVENTION WITHOUT FERTILIZERS:"

"Remove infected leaves and plants."
"Avoid overhead irrigation."
"Practice crop rotation."
"Maintain good spacing for airflow."
"Keep field clean from debris."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium for plant strength."
"Spray Copper or Sulfur-based fungicides."
"Apply Zinc and Magnesium micronutrients."
"Bio-fertilizers improve plant resistance.", 

          '***Tomato___Tomato_Yellow_Leaf_Curl_Virus***'
"Tomato Yellow Leaf Curl Virus is a serious viral disease of tomato caused by Begomovirus. It is mainly spread by the whitefly (Bemisia tabaci). The disease causes major yield loss by affecting plant growth and fruit production."

"###CAUSE OF YELLOW LEAF CURL VIRUS:"

"Caused by Begomovirus."
"Spread by whitefly (Bemisia tabaci)."
"Symptoms include yellowing, upward curling of leaves, stunted growth, and small fruits."
"Plants become weak and produce very low yield."
"Common in warm and dry conditions."

"###PREVENTION WITHOUT FERTILIZERS:"

"Control whiteflies using sticky traps."
"Remove infected plants immediately."
"Use insect-proof nets in nurseries."
"Keep field weed-free."
"Practice crop rotation."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium to improve plant strength."
"Apply micronutrients like Zinc and Magnesium."
"Use neem-based sprays or mild insecticides for whiteflies."
"Bio-fertilizers improve plant health and resistance.", 
            '***Tomato___Tomato_mosaic_virus***'
"Tomato Mosaic Virus is a viral disease of tomato caused by Tobamovirus. It spreads easily through contaminated tools, hands, seeds, and plant contact. The disease reduces plant growth, fruit quality, and overall yield."

"###CAUSE OF TOMATO MOSAIC VIRUS:"

"Caused by Tobamovirus."
"Spreads through infected seeds, tools, hands, and plant contact."
"Symptoms include mottled green and yellow leaves, leaf curling, and stunted growth."
"Fruits may become deformed and uneven."
"Common in warm conditions and greenhouse crops."

"###PREVENTION WITHOUT FERTILIZERS:"

"Use virus-free seeds."
"Disinfect tools regularly."
"Remove infected plants immediately."
"Avoid touching healthy plants after infected ones."
"Maintain field hygiene."

"###PREVENTION WITH FERTILIZERS (AVAILABLE IN SHOPS):"

"Apply balanced NPK fertilizers."
"Use Potassium to improve plant strength."
"Apply Zinc and Magnesium micronutrients."
"Use organic bio-fertilizers to improve plant health."
"There is no cure, only prevention and management.",
                    '***Tomato___healthy***']
        st.success("NAME OF THE DISEASE : {}".format(class_name[result_index]))
