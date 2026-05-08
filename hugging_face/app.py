import gradio as gr
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

CONFIDENCE_THRESHOLD = 0.65
IMG_SIZE             = 224

# ============================================================
# DISEASE REMEDY DATABASE
# ============================================================

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display_name": "Apple — Apple Scab",
        "plant": "Apple",
        "status": "diseased",
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. It is one of the most common and destructive diseases of apple trees worldwide, thriving in cool and wet spring conditions.",
        "symptoms": "Olive-green to brown velvety spots on leaves and fruit. Leaves may yellow and drop early. Fruit develops raised, dark, corky lesions that crack as fruit matures.",
        "remedies": [
            "Apply fungicides containing captan, myclobutanil, or mancozeb at 7–10 day intervals starting at bud break.",
            "Rake and destroy fallen leaves in autumn to remove overwintering fungal spores.",
            "Prune trees to improve air circulation and reduce humidity within the canopy.",
            "Plant resistant apple varieties such as Liberty, Freedom, or Enterprise for future seasons.",
            "Avoid overhead irrigation — water at the base of the tree to keep foliage dry."
        ]
    },
    "Apple___Black_rot": {
        "display_name": "Apple — Black Rot",
        "plant": "Apple",
        "status": "diseased",
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It affects fruit, leaves, and bark, and can cause significant crop losses if left unmanaged.",
        "symptoms": "Circular brown lesions with purple borders on leaves. Fruit shows brown rot starting from the blossom end, eventually turning black and shrivelled. Cankers appear on bark.",
        "remedies": [
            "Remove and destroy mummified fruit and dead wood — these are primary sources of infection.",
            "Apply copper-based fungicides or captan during the growing season.",
            "Prune out all dead and cankered wood during dry weather and seal cuts with pruning paint.",
            "Maintain tree vigor through proper fertilisation and irrigation — stressed trees are more susceptible.",
            "Ensure good orchard sanitation by clearing debris around the base of trees."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "display_name": "Apple — Cedar Apple Rust",
        "plant": "Apple",
        "status": "diseased",
        "description": "Cedar apple rust is caused by Gymnosporangium juniperi-virginianae, a fungus that requires two host plants — apple and eastern red cedar or juniper — to complete its life cycle.",
        "symptoms": "Bright orange-yellow spots on upper leaf surfaces in spring, with tube-like structures on the underside. Infected fruit develops orange lesions and may become misshapen.",
        "remedies": [
            "Apply myclobutanil or propiconazole fungicides from pink bud stage through petal fall.",
            "Remove nearby juniper and cedar trees if possible — they are the alternate host for the fungus.",
            "Plant rust-resistant apple varieties such as Redfree, Williams Pride, or Pristine.",
            "Spray protective fungicides every 7–10 days during wet spring conditions.",
            "Inspect juniper trees for orange gelatinous galls in spring and remove them promptly."
        ]
    },
    "Apple___healthy": {
        "display_name": "Apple — Healthy",
        "plant": "Apple",
        "status": "healthy",
        "description": "Your apple plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Continue regular monitoring — inspect leaves, fruit, and bark weekly for early signs of disease.",
            "Maintain a balanced fertilisation programme with nitrogen, phosphorus, and potassium.",
            "Prune annually during dormancy to improve air circulation and light penetration.",
            "Apply a preventive dormant oil spray in late winter to control overwintering pests.",
            "Ensure consistent watering, especially during fruit development, to prevent stress."
        ]
    },
    "Blueberry___healthy": {
        "display_name": "Blueberry — Healthy",
        "plant": "Blueberry",
        "status": "healthy",
        "description": "Your blueberry plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Maintain soil pH between 4.5 and 5.5 — blueberries require acidic soil to thrive.",
            "Mulch around plants with pine bark or sawdust to retain moisture and suppress weeds.",
            "Prune out old and weak canes each year to encourage vigorous new growth.",
            "Apply balanced fertiliser formulated for acid-loving plants in early spring.",
            "Monitor for common pests such as spotted wing drosophila and mummy berry disease."
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display_name": "Cherry — Powdery Mildew",
        "plant": "Cherry",
        "status": "diseased",
        "description": "Powdery mildew on cherry is caused by Podosphaera clandestina. It thrives in warm, dry conditions with high humidity, particularly affecting young tissue.",
        "symptoms": "White powdery coating on young leaves, shoots, and fruit. Infected leaves may curl, distort, or drop early. Severely affected shoots may be stunted.",
        "remedies": [
            "Apply sulphur-based fungicides or potassium bicarbonate as soon as symptoms appear.",
            "Use systemic fungicides such as myclobutanil or trifloxystrobin for severe infections.",
            "Prune out heavily infected shoots and dispose of them away from the orchard.",
            "Improve air circulation by thinning the canopy and avoiding overcrowding of plants.",
            "Avoid excessive nitrogen fertilisation which promotes the lush soft growth mildew prefers."
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "display_name": "Cherry — Healthy",
        "plant": "Cherry",
        "status": "healthy",
        "description": "Your cherry plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Monitor regularly for brown rot, which is the most common cherry disease.",
            "Apply a preventive fungicide programme around flowering time in wet seasons.",
            "Net trees to protect fruit from birds, which can spread disease by damaging fruit skins.",
            "Prune in dry weather to minimise infection risk through pruning wounds.",
            "Ensure adequate potassium levels to improve fruit quality and disease resistance."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display_name": "Corn — Gray Leaf Spot",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "description": "Gray leaf spot is caused by the fungus Cercospora zeae-maydis. It thrives in warm, humid conditions with extended dew periods and is one of the most yield-limiting corn diseases globally.",
        "symptoms": "Long, rectangular grey to tan lesions running parallel to leaf veins. Severely infected leaves may die, reducing photosynthetic area significantly.",
        "remedies": [
            "Plant resistant or tolerant corn hybrids — this is the most effective long-term management strategy.",
            "Apply fungicides containing strobilurin or triazole at VT (tasselling) stage if infection is severe.",
            "Rotate crops — avoid planting corn on the same field in consecutive years.",
            "Till crop residue after harvest to reduce the amount of overwintering fungal material.",
            "Improve field drainage and avoid fields with poor air circulation that stay humid."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "display_name": "Corn — Common Rust",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "description": "Common rust is caused by Puccinia sorghi. It spreads rapidly under cool, moist conditions and can significantly reduce yield if infection occurs early in the season.",
        "symptoms": "Small, circular to elongated brick-red pustules scattered on both leaf surfaces. Heavy infections cause leaves to yellow and die.",
        "remedies": [
            "Plant rust-resistant corn hybrids as the primary defence strategy.",
            "Apply fungicides containing azoxystrobin, pyraclostrobin, or propiconazole at early stages of infection.",
            "Scout fields regularly from tasselling onwards and act quickly if pustule counts are high.",
            "Avoid late planting dates — early-planted corn often escapes the peak rust infection period.",
            "Ensure good plant nutrition, particularly adequate potassium, to enhance natural resistance."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display_name": "Corn — Northern Leaf Blight",
        "plant": "Corn (Maize)",
        "status": "diseased",
        "description": "Northern leaf blight is caused by Exserohilum turcicum. It is particularly damaging when infection occurs before or during tasselling.",
        "symptoms": "Long, cigar-shaped grey-green to tan lesions, typically 2.5–15 cm in length. Lesions appear first on lower leaves and progress upward.",
        "remedies": [
            "Plant hybrids with partial resistance to northern leaf blight — the most practical management tool.",
            "Apply strobilurin or triazole fungicides around the VT/R1 growth stage if weather is favourable.",
            "Rotate corn with soybeans or other non-host crops to reduce inoculum in crop debris.",
            "Incorporate crop residues after harvest to speed decomposition of infected plant material.",
            "Avoid excessive plant populations which increase humidity and disease spread within the canopy."
        ]
    },
    "Corn_(maize)___healthy": {
        "display_name": "Corn — Healthy",
        "plant": "Corn (Maize)",
        "status": "healthy",
        "description": "Your corn plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Scout fields weekly from emergence through grain fill to catch problems early.",
            "Maintain balanced soil fertility — test soil annually and adjust nutrient applications accordingly.",
            "Rotate with soybeans to break disease and pest cycles.",
            "Ensure proper plant spacing to promote good air circulation and reduce foliar disease risk.",
            "Monitor for corn rootworm and European corn borer — early pest management prevents yield loss."
        ]
    },
    "Grape___Black_rot": {
        "display_name": "Grape — Black Rot",
        "plant": "Grape",
        "status": "diseased",
        "description": "Black rot is caused by the fungus Guignardia bidwellii. It can destroy entire crops in wet seasons and is one of the most serious diseases of grapes in humid climates.",
        "symptoms": "Circular tan-brown lesions with dark borders on leaves. Infected berries turn brown, then black and shrivelled into hard mummified fruit.",
        "remedies": [
            "Apply fungicides containing myclobutanil, mancozeb, or captan from budbreak through veraison.",
            "Remove and destroy all mummified fruit — they are the primary source of inoculum the following season.",
            "Prune to open the canopy and improve air circulation — this reduces humidity and drying time.",
            "Train vines on a trellis system to keep fruit and foliage off the ground.",
            "Begin fungicide programme early — once fruit is infected, it cannot be cured."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "display_name": "Grape — Esca (Black Measles)",
        "plant": "Grape",
        "status": "diseased",
        "description": "Esca is a complex grapevine trunk disease caused by several wood-rotting fungi. It is a chronic, progressive disease with no cure.",
        "symptoms": "Interveinal tiger-stripe yellowing or reddening of leaves. Berries develop small dark spots surrounded by purple halos. In acute form, the vine may suddenly wilt and die in summer.",
        "remedies": [
            "There is no cure once a vine is infected — focus on prevention and managing spread.",
            "Prune during dry weather and protect pruning wounds immediately with fungicidal paste.",
            "Remove and burn severely affected vines to prevent spread to healthy plants.",
            "Delay pruning as late as possible in the dormant season to reduce wound infection risk.",
            "Replant with certified disease-free nursery stock and sterilise pruning tools between vines."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display_name": "Grape — Leaf Blight (Isariopsis Leaf Spot)",
        "plant": "Grape",
        "status": "diseased",
        "description": "Grape leaf blight caused by Isariopsis clavispora appears later in the season and causes premature defoliation, weakening vines going into dormancy.",
        "symptoms": "Irregular dark brown to black lesions on older leaves, often with a yellow halo. Severely affected leaves drop prematurely.",
        "remedies": [
            "Apply copper-based fungicides or mancozeb during the growing season as a protective measure.",
            "Remove fallen infected leaves and debris from around vines to reduce overwintering inoculum.",
            "Maintain good canopy management through shoot positioning and leaf removal to improve air flow.",
            "Ensure vines are not under stress from drought or nutrient deficiency.",
            "Apply a balanced fertiliser programme to maintain vine vigour through the growing season."
        ]
    },
    "Grape___healthy": {
        "display_name": "Grape — Healthy",
        "plant": "Grape",
        "status": "healthy",
        "description": "Your grapevine appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Maintain a regular preventive fungicide programme for downy mildew and powdery mildew.",
            "Train shoots and remove excess foliage around the fruit zone to improve air circulation.",
            "Monitor soil moisture — grapevines prefer well-drained soils and are sensitive to waterlogging.",
            "Test soil and petioles annually to ensure balanced nutrition, particularly potassium and magnesium.",
            "Sterilise all pruning and harvesting tools between vines to prevent disease transmission."
        ]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display_name": "Orange — Huanglongbing (Citrus Greening)",
        "plant": "Orange",
        "status": "diseased",
        "description": "Huanglongbing (HLB), or citrus greening, is caused by the bacterium Candidatus Liberibacter asiaticus and spread by the Asian citrus psyllid. It is the most devastating citrus disease in the world with no cure.",
        "symptoms": "Asymmetric or blotchy yellowing of leaves. Fruit remains small, green, lopsided, and bitter. Trees show progressive decline over several years.",
        "remedies": [
            "There is currently no cure for HLB — infected trees should be removed and destroyed.",
            "Control the Asian citrus psyllid vector aggressively using systemic insecticides such as imidacloprid.",
            "Plant certified HLB-free nursery stock only — never source trees from unverified suppliers.",
            "Establish insect exclusion screens in nurseries to prevent psyllid access to young trees.",
            "Apply nutritional sprays containing micronutrients to slow the decline of mildly affected trees."
        ]
    },
    "Peach___Bacterial_spot": {
        "display_name": "Peach — Bacterial Spot",
        "plant": "Peach",
        "status": "diseased",
        "description": "Bacterial spot is caused by Xanthomonas arboricola pv. pruni. It is one of the most destructive diseases of peach, causing significant defoliation and fruit blemishing in warm, wet conditions.",
        "symptoms": "Small, water-soaked spots on leaves that turn angular and purple-brown with yellow halos. Spots may fall out leaving a shot-hole appearance.",
        "remedies": [
            "Apply copper-based bactericides from bud swell through petal fall at 7–10 day intervals.",
            "Plant resistant or tolerant peach varieties — resistance is the most reliable long-term strategy.",
            "Prune to improve air circulation and avoid working in the orchard when foliage is wet.",
            "Avoid high-nitrogen fertilisation which produces lush, susceptible new growth.",
            "Apply oxytetracycline sprays during bloom in severe cases — consult local regulations."
        ]
    },
    "Peach___healthy": {
        "display_name": "Peach — Healthy",
        "plant": "Peach",
        "status": "healthy",
        "description": "Your peach plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Apply a dormant copper spray before bud swell each year as a preventive measure.",
            "Thin fruit to 15–20 cm apart after natural drop to improve fruit size and air circulation.",
            "Monitor for peach leaf curl — a common fungal disease requiring early spring treatment.",
            "Maintain soil fertility with balanced fertiliser and avoid excess nitrogen.",
            "Prune annually to maintain an open vase shape that promotes light penetration and air movement."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "display_name": "Bell Pepper — Bacterial Spot",
        "plant": "Bell Pepper",
        "status": "diseased",
        "description": "Bacterial spot of pepper is caused by Xanthomonas campestris pv. vesicatoria. It can cause severe defoliation and fruit loss in warm, wet weather.",
        "symptoms": "Small, water-soaked spots on leaves that enlarge and turn brown with yellow halos. Fruit develops raised, brown, scab-like spots that reduce quality.",
        "remedies": [
            "Apply copper-based bactericides combined with mancozeb at 5–7 day intervals during wet weather.",
            "Use certified disease-free transplants and treat seeds with hot water (50°C for 25 minutes).",
            "Rotate peppers with non-solanaceous crops for at least 2 years to reduce soil inoculum.",
            "Avoid working in the field when plants are wet — the bacteria spread easily through water and contact.",
            "Remove and destroy heavily infected plant material — do not compost diseased tissue."
        ]
    },
    "Pepper,_bell___healthy": {
        "display_name": "Bell Pepper — Healthy",
        "plant": "Bell Pepper",
        "status": "healthy",
        "description": "Your bell pepper plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Stake plants early to prevent fruit and foliage from touching the soil.",
            "Water at the base of plants — avoid overhead irrigation which promotes bacterial and fungal diseases.",
            "Apply balanced fertiliser with adequate calcium to prevent blossom end rot.",
            "Monitor for aphids and thrips which can transmit viral diseases.",
            "Rotate with non-solanaceous crops each season to maintain soil health."
        ]
    },
    "Potato___Early_blight": {
        "display_name": "Potato — Early Blight",
        "plant": "Potato",
        "status": "diseased",
        "description": "Early blight is caused by the fungus Alternaria solani. It typically affects older and stressed plants, favoured by warm temperatures with alternating wet and dry periods.",
        "symptoms": "Dark brown to black circular lesions with concentric rings forming a target-board pattern. Lower, older leaves are affected first, progressing upwards.",
        "remedies": [
            "Apply fungicides containing chlorothalonil, mancozeb, or azoxystrobin at first sign of symptoms.",
            "Ensure adequate fertilisation — nitrogen-deficient plants are more susceptible to early blight.",
            "Use certified disease-free seed potatoes and plant resistant varieties where available.",
            "Rotate potatoes with non-solanaceous crops for 2–3 years to reduce soilborne inoculum.",
            "Remove and destroy infected plant debris after harvest — do not leave in the field."
        ]
    },
    "Potato___Late_blight": {
        "display_name": "Potato — Late Blight",
        "plant": "Potato",
        "status": "diseased",
        "description": "Late blight is caused by Phytophthora infestans — the same pathogen responsible for the Irish Potato Famine. It can devastate entire crops within days under cool, wet conditions.",
        "symptoms": "Water-soaked, pale green lesions on leaves that rapidly turn dark brown to black. White fluffy mould visible on the underside of leaves in humid conditions.",
        "remedies": [
            "Act immediately — late blight spreads extremely rapidly. Apply fungicides containing cymoxanil, metalaxyl, or mandipropamid.",
            "Apply protective fungicides before infection occurs when weather conditions are cool and wet.",
            "Destroy infected haulm before harvesting to prevent tuber infection.",
            "Plant certified blight-free seed potatoes and resistant varieties such as Sarpo Mira.",
            "Avoid irrigation in the evening — wet foliage overnight dramatically increases infection risk."
        ]
    },
    "Potato___healthy": {
        "display_name": "Potato — Healthy",
        "plant": "Potato",
        "status": "healthy",
        "description": "Your potato plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Monitor weather forecasts — apply preventive late blight fungicides before prolonged cool, wet periods.",
            "Hill up soil around stems as plants grow to protect tubers from greening and blight.",
            "Water in the morning so foliage dries quickly during the day.",
            "Use certified seed potatoes — never plant potatoes from the previous season's grocery store stock.",
            "Maintain good weed control to improve air circulation and reduce humidity around plants."
        ]
    },
    "Raspberry___healthy": {
        "display_name": "Raspberry — Healthy",
        "plant": "Raspberry",
        "status": "healthy",
        "description": "Your raspberry plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Remove all fruited canes after harvest — this prevents disease buildup and encourages new growth.",
            "Tie canes to a support wire system to keep them off the ground and improve air flow.",
            "Apply balanced fertiliser in early spring before growth begins.",
            "Monitor for raspberry cane blight and botrytis — both are common in wet seasons.",
            "Mulch around the base of plants to conserve moisture and suppress weeds."
        ]
    },
    "Soybean___healthy": {
        "display_name": "Soybean — Healthy",
        "plant": "Soybean",
        "status": "healthy",
        "description": "Your soybean plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Scout fields regularly for sudden death syndrome, soybean cyst nematode, and frogeye leaf spot.",
            "Rotate soybeans with corn or small grains to prevent pathogen and pest buildup.",
            "Use high-quality certified seed treated with fungicide and rhizobium inoculant.",
            "Avoid compaction by limiting field traffic during wet conditions.",
            "Apply potassium and phosphorus according to soil test results to maintain plant vigour."
        ]
    },
    "Squash___Powdery_mildew": {
        "display_name": "Squash — Powdery Mildew",
        "plant": "Squash",
        "status": "diseased",
        "description": "Powdery mildew on squash is caused by Podosphaera xanthii. Unlike most fungal diseases, it thrives in warm, dry conditions with moderate humidity.",
        "symptoms": "White, powdery circular patches on the upper leaf surface that gradually cover entire leaves. Affected leaves may yellow, curl, and die.",
        "remedies": [
            "Apply sulphur-based fungicides, potassium bicarbonate, or neem oil at first signs of infection.",
            "Use systemic fungicides such as myclobutanil or trifloxystrobin for established infections.",
            "Plant resistant squash varieties — many modern varieties have good powdery mildew tolerance.",
            "Avoid excessive nitrogen which promotes the soft lush growth that mildew colonises most easily.",
            "Space plants adequately and remove old infected leaves to improve air circulation and slow spread."
        ]
    },
    "Strawberry___Leaf_scorch": {
        "display_name": "Strawberry — Leaf Scorch",
        "plant": "Strawberry",
        "status": "diseased",
        "description": "Leaf scorch is caused by the fungus Diplocarpon earlianum. It is one of the most common foliar diseases of strawberry and can cause significant reduction in plant vigour and yield.",
        "symptoms": "Small, irregular dark purple to reddish-purple spots on the upper leaf surface. Severely affected leaves have a scorched or burned appearance as lesions merge.",
        "remedies": [
            "Apply fungicides containing captan or myclobutanil from early spring through the growing season.",
            "Remove and destroy infected leaves and runners — do not leave on the soil surface.",
            "Renovate strawberry beds after harvest by mowing foliage and removing debris to reduce inoculum.",
            "Plant resistant varieties and use certified disease-free planting material.",
            "Avoid overhead irrigation — water at the base of plants and ensure good drainage."
        ]
    },
    "Strawberry___healthy": {
        "display_name": "Strawberry — Healthy",
        "plant": "Strawberry",
        "status": "healthy",
        "description": "Your strawberry plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Mulch around plants with straw to keep fruit clean, retain moisture, and reduce splash dispersal of pathogens.",
            "Remove runners promptly unless increasing plant population — they divert energy from fruit production.",
            "Apply balanced fertiliser after harvest to build plant reserves for the following season.",
            "Monitor for botrytis (grey mould) during flowering — it is the most damaging strawberry disease.",
            "Renovate the bed after fruiting by cutting back foliage to encourage vigorous regrowth."
        ]
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato — Bacterial Spot",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Bacterial spot of tomato is caused by Xanthomonas vesicatoria and related species. It is spread by rain splash, wind-driven rain, and contaminated tools.",
        "symptoms": "Small, dark brown water-soaked spots on leaves with yellow halos. Fruit develops raised, brown, scab-like spots. Severe infections cause significant defoliation.",
        "remedies": [
            "Apply copper-based bactericides combined with mancozeb at 5–7 day intervals during wet weather.",
            "Use disease-free transplants or treat seeds with hot water at 50°C for 25 minutes.",
            "Rotate tomatoes with non-solanaceous crops for at least 2–3 years.",
            "Stake plants and remove lower leaves to improve air circulation and reduce splash dispersal.",
            "Avoid working in the garden when foliage is wet to prevent mechanical spread."
        ]
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato — Early Blight",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Early blight of tomato is caused by Alternaria solani. It typically affects older leaves first and progresses upward, causing significant defoliation.",
        "symptoms": "Dark brown to black circular lesions with concentric rings giving a target-board appearance. Lower, older leaves are affected first.",
        "remedies": [
            "Apply fungicides containing chlorothalonil, mancozeb, or copper hydroxide at 7–10 day intervals.",
            "Remove and destroy infected lower leaves to slow upward spread of the disease.",
            "Stake and tie plants to keep foliage and fruit off the ground.",
            "Rotate tomatoes with non-solanaceous crops and avoid planting in the same location each year.",
            "Ensure adequate calcium and consistent watering to maintain plant vigour and resistance."
        ]
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato — Late Blight",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Late blight of tomato is caused by Phytophthora infestans. It can destroy entire plantings within days in cool, wet conditions.",
        "symptoms": "Water-soaked, irregular pale green to brown lesions on leaves that rapidly expand and turn black. White fluffy mould on the underside of leaves in humid conditions.",
        "remedies": [
            "Act immediately — apply fungicides containing cymoxanil, mandipropamid, or chlorothalonil without delay.",
            "Remove and destroy all infected plant material — bag and bin it, do not compost.",
            "Apply preventive fungicides before cool, wet weather periods when infection is most likely.",
            "Avoid wetting foliage during irrigation — use drip irrigation at the base of plants.",
            "Plant late blight resistant tomato varieties for future seasons."
        ]
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato — Leaf Mold",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Leaf mould is caused by the fungus Passalora fulva. It primarily affects tomatoes grown in greenhouses or polytunnels where humidity is high.",
        "symptoms": "Pale green to yellow spots on the upper leaf surface with olive-green to grey velvety mould on the corresponding underside. Infected leaves eventually turn yellow and drop.",
        "remedies": [
            "Reduce humidity by improving ventilation in greenhouses — open vents and doors during the day.",
            "Apply fungicides containing chlorothalonil, mancozeb, or copper hydroxide at first sign of symptoms.",
            "Remove and destroy infected leaves promptly to reduce spore load in the environment.",
            "Avoid overhead irrigation and wetting foliage — water at the base of plants.",
            "Plant resistant tomato varieties — many modern glasshouse varieties have good leaf mould resistance."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato — Septoria Leaf Spot",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Septoria leaf spot is caused by the fungus Septoria lycopersici. It causes progressive defoliation from the bottom of the plant upwards.",
        "symptoms": "Numerous small, circular spots with white or grey centres and dark brown borders on lower leaves. Infected leaves yellow and drop progressively.",
        "remedies": [
            "Apply fungicides containing chlorothalonil, mancozeb, or copper at 7–10 day intervals during wet weather.",
            "Remove infected lower leaves as soon as symptoms appear to slow upward progression.",
            "Mulch around plants to prevent soil splash which spreads spores to lower leaves.",
            "Stake plants to keep foliage off the ground and improve air circulation.",
            "Rotate tomatoes and avoid planting in the same bed more than once every 3 years."
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "display_name": "Tomato — Spider Mites",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that thrive in hot, dry conditions. They pierce leaf cells and extract their contents, causing rapid decline if uncontrolled.",
        "symptoms": "Fine yellow stippling on the upper leaf surface giving a dusty or bronzed appearance. Fine webbing visible on the underside of leaves and between stems.",
        "remedies": [
            "Apply miticides containing abamectin, bifenazate, or spiromesifen — rotate between modes of action to prevent resistance.",
            "Introduce predatory mites such as Phytoseiulus persimilis as a biological control in greenhouse settings.",
            "Spray plants forcefully with water to dislodge mites — this can reduce populations significantly.",
            "Maintain adequate soil moisture — spider mites thrive in drought-stressed plants.",
            "Remove heavily infested leaves and avoid using broad-spectrum insecticides which kill natural predators."
        ]
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato — Target Spot",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Target spot is caused by the fungus Corynespora cassiicola. It is increasingly common in tropical and subtropical regions and can cause significant defoliation and fruit loss.",
        "symptoms": "Brown circular lesions with concentric rings (target pattern) on leaves, stems, and fruit. Fruit develops sunken, dark brown spots that can lead to rotting.",
        "remedies": [
            "Apply fungicides containing chlorothalonil, azoxystrobin, or difenoconazole at first sign of symptoms.",
            "Remove infected leaves and fruit promptly to reduce spore production.",
            "Improve air circulation through pruning, staking, and adequate plant spacing.",
            "Avoid overhead watering — use drip irrigation to keep foliage dry.",
            "Rotate with non-solanaceous crops and incorporate crop debris thoroughly after harvest."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato — Yellow Leaf Curl Virus",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Tomato Yellow Leaf Curl Virus (TYLCV) is transmitted exclusively by the silverleaf whitefly. It is one of the most damaging tomato viruses worldwide.",
        "symptoms": "Upward curling and yellowing of leaf edges on young leaves. Plants are severely stunted. Flowers drop without setting fruit.",
        "remedies": [
            "There is no cure once a plant is infected — remove and destroy infected plants immediately.",
            "Control whitefly populations aggressively using systemic insecticides such as imidacloprid or thiamethoxam.",
            "Use reflective silver mulch on the soil surface — this disorients whiteflies and reduces landing rates.",
            "Install yellow sticky traps to monitor and reduce adult whitefly populations.",
            "Plant TYLCV-resistant tomato varieties — many commercial varieties now carry resistance genes."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato — Mosaic Virus",
        "plant": "Tomato",
        "status": "diseased",
        "description": "Tomato mosaic virus (ToMV) is a highly stable and persistent virus spread primarily through contact with contaminated tools, hands, and infected plant material.",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves. Leaves may be distorted or curled. Plants may be stunted.",
        "remedies": [
            "There is no cure — remove and destroy infected plants to prevent spread.",
            "Wash hands thoroughly with soap before handling plants, especially after smoking.",
            "Sterilise all tools with 10% bleach solution or 70% alcohol between uses.",
            "Plant resistant tomato varieties carrying the Tm-2 resistance gene.",
            "Control aphids which may act as secondary vectors and avoid unnecessary plant handling."
        ]
    },
    "Tomato___healthy": {
        "display_name": "Tomato — Healthy",
        "plant": "Tomato",
        "status": "healthy",
        "description": "Your tomato plant appears healthy with no visible signs of disease.",
        "symptoms": "No symptoms detected.",
        "remedies": [
            "Water consistently at the base of plants — irregular watering causes blossom end rot and fruit cracking.",
            "Apply calcium-rich fertiliser to prevent blossom end rot, which is common in rapidly growing plants.",
            "Remove suckers (side shoots) regularly to maintain a clean, well-ventilated plant structure.",
            "Monitor weekly for early signs of blight, septoria, and spider mites — early detection saves crops.",
            "Stake or cage plants early to keep fruit and foliage off the ground."
        ]
    },
}


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")
try:
    processor_hf = AutoImageProcessor.from_pretrained(".")
    model        = AutoModelForImageClassification.from_pretrained(".")
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model        = None
    processor_hf = None

IMAGE_MEAN = processor_hf.image_mean if processor_hf else [0.485, 0.456, 0.406]
IMAGE_STD  = processor_hf.image_std  if processor_hf else [0.229, 0.224, 0.225]


# ============================================================
# TTA TRANSFORMS
# ============================================================

tta_transforms = [
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ]),
]


# ============================================================
# INFERENCE
# ============================================================

def predict(image: Image.Image):
    if model is None:
        return build_error_html("Model failed to load. Please check the Space logs.")

    if image is None:
        return build_error_html("Please upload a leaf image to begin analysis.")

    try:
        image      = image.convert("RGB")
        probs_list = []

        with torch.no_grad():
            for tfm in tta_transforms:
                tensor = tfm(image).unsqueeze(0)
                logits = model(pixel_values=tensor).logits
                probs  = torch.softmax(logits, dim=1).squeeze().numpy()
                probs_list.append(probs)

        avg_probs  = np.mean(probs_list, axis=0)
        confidence = float(np.max(avg_probs))
        pred_idx   = int(np.argmax(avg_probs))
        pred_label = model.config.id2label[pred_idx]

        top3_idx = np.argsort(avg_probs)[::-1][:3]
        top3     = [(model.config.id2label[i], float(avg_probs[i]))
                    for i in top3_idx]

        if confidence < CONFIDENCE_THRESHOLD:
            return build_uncertain_html(top3)

        info = DISEASE_INFO.get(pred_label)
        if info is None:
            return build_error_html(
                f"Disease detected ({pred_label}) but no remedy information found."
            )

        return build_result_html(info, confidence, top3)

    except Exception as e:
        return build_error_html(f"An error occurred during analysis: {str(e)}")


# ============================================================
# HTML BUILDERS
# ============================================================

def confidence_label(conf: float) -> str:
    if conf >= 0.85:
        return '<span class="conf-high">High Confidence</span>'
    elif conf >= 0.65:
        return '<span class="conf-medium">Medium Confidence</span>'
    else:
        return '<span class="conf-low">Low Confidence</span>'


def build_result_html(info: dict, confidence: float, top3: list) -> str:
    is_healthy   = info["status"] == "healthy"
    status_class = "status-healthy" if is_healthy else "status-diseased"
    status_text  = "✓ Healthy Plant" if is_healthy else "⚠ Disease Detected"

    remedies_html = "".join(
        f'<li class="remedy-item">{r}</li>' for r in info["remedies"]
    )

    top3_html = "".join(
        f'<div class="top3-item">'
        f'<span class="top3-name">{DISEASE_INFO.get(label, {}).get("display_name", label)}</span>'
        f'<div class="top3-bar-wrap"><div class="top3-bar" style="width:{c*100:.0f}%"></div></div>'
        f'<span class="top3-pct">{c*100:.1f}%</span>'
        f'</div>'
        for label, c in top3
    )

    symptoms_section = (
        f'<div class="section">'
        f'<div class="section-title">🔍 Symptoms</div>'
        f'<p class="section-text">{info["symptoms"]}</p>'
        f'</div>'
    ) if not is_healthy else ""

    treatment_title = "💚 Care Recommendations" if is_healthy else "💊 Treatment & Management"

    return f"""
    <div class="result-card">
        <div class="result-header {status_class}">
            <div class="status-badge">{status_text}</div>
            <div class="disease-name">{info['display_name']}</div>
            <div class="conf-badge">{confidence_label(confidence)}</div>
        </div>
        <div class="result-body">
            <div class="section">
                <div class="section-title">📋 Description</div>
                <p class="section-text">{info['description']}</p>
            </div>
            {symptoms_section}
            <div class="section">
                <div class="section-title">{treatment_title}</div>
                <ul class="remedy-list">{remedies_html}</ul>
            </div>
            <div class="section">
                <div class="section-title">📊 Top Predictions</div>
                <div class="top3-container">{top3_html}</div>
            </div>
        </div>
    </div>
    """


def build_uncertain_html(top3: list) -> str:
    top3_html = "".join(
        f'<div class="top3-item">'
        f'<span class="top3-name">{DISEASE_INFO.get(label, {}).get("display_name", label)}</span>'
        f'<div class="top3-bar-wrap"><div class="top3-bar" style="width:{c*100:.0f}%"></div></div>'
        f'<span class="top3-pct">{c*100:.1f}%</span>'
        f'</div>'
        for label, c in top3
    )
    return f"""
    <div class="result-card">
        <div class="result-header status-uncertain">
            <div class="status-badge">⚡ Uncertain Result</div>
            <div class="disease-name">Unable to identify with confidence</div>
        </div>
        <div class="result-body">
            <div class="section">
                <div class="section-title">Suggestions to improve accuracy</div>
                <ul class="remedy-list">
                    <li class="remedy-item">Retake the photo in bright natural daylight.</li>
                    <li class="remedy-item">Focus closely on the affected leaf — fill the frame with the leaf.</li>
                    <li class="remedy-item">Ensure the image is in focus and not blurry.</li>
                    <li class="remedy-item">Avoid strong shadows or glare on the leaf surface.</li>
                    <li class="remedy-item">Try uploading a different affected leaf from the same plant.</li>
                </ul>
            </div>
            <div class="section">
                <div class="section-title">📊 Top Predictions</div>
                <div class="top3-container">{top3_html}</div>
            </div>
        </div>
    </div>
    """


def build_error_html(message: str) -> str:
    return f"""
    <div class="result-card">
        <div class="result-header status-uncertain">
            <div class="status-badge">❌ Error</div>
            <div class="disease-name">{message}</div>
        </div>
    </div>
    """


# ============================================================
# CSS
# ============================================================

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f0f4f0 !important;
}
.app-header {
    background: linear-gradient(135deg, #1a4a1a 0%, #2d7a2d 50%, #1a4a1a 100%);
    padding: 2.5rem 2rem 2rem; text-align: center;
    border-radius: 0 0 24px 24px; margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
.app-title { font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin-bottom: 0.4rem; }
.app-subtitle { font-size: 1rem; color: #a8d8a8; font-weight: 400; }
.app-leaf-icon { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; }
.upload-section { background: #ffffff; border: 2px dashed #4a9a4a; border-radius: 16px; padding: 1.5rem; text-align: center; }
.analyse-btn {
    background: #2d7a2d !important; color: white !important; border: none !important;
    border-radius: 12px !important; font-size: 1.05rem !important; font-weight: 600 !important;
    padding: 0.75rem 2rem !important; cursor: pointer !important;
    transition: all 0.2s !important; width: 100% !important; margin-top: 1rem !important;
}
.analyse-btn:hover { background: #1a5c1a !important; transform: translateY(-1px) !important; box-shadow: 0 4px 15px rgba(45,122,45,0.3) !important; }
.result-card { background: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); font-family: 'Segoe UI', system-ui, sans-serif; }
.result-header { padding: 1.5rem 1.75rem; color: white; }
.status-diseased  { background: linear-gradient(135deg, #b85c00, #e07020); }
.status-healthy   { background: linear-gradient(135deg, #1a6b1a, #2d9a2d); }
.status-uncertain { background: linear-gradient(135deg, #5c5c1a, #8a8a2d); }
.status-badge { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 0.5rem; }
.disease-name { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem; line-height: 1.2; }
.conf-badge { font-size: 0.85rem; opacity: 0.85; }
.conf-high   { color: #90ee90; font-weight: 600; }
.conf-medium { color: #ffd700; font-weight: 600; }
.conf-low    { color: #ffa07a; font-weight: 600; }
.result-body { padding: 1.5rem 1.75rem; }
.section { margin-bottom: 1.4rem; }
.section:last-child { margin-bottom: 0; }
.section-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #2d7a2d; margin-bottom: 0.6rem; border-bottom: 2px solid #e8f5e8; padding-bottom: 0.4rem; }
.section-text { font-size: 0.95rem; color: #3a3a3a; line-height: 1.6; }
.remedy-list { list-style: none; padding: 0; }
.remedy-item { font-size: 0.9rem; color: #3a3a3a; line-height: 1.5; padding: 0.5rem 0 0.5rem 1.5rem; border-bottom: 1px solid #f0f0f0; position: relative; }
.remedy-item:last-child { border-bottom: none; }
.remedy-item::before { content: "→"; position: absolute; left: 0; color: #2d7a2d; font-weight: 700; }
.top3-container { display: flex; flex-direction: column; gap: 0.5rem; }
.top3-item { display: flex; align-items: center; gap: 0.75rem; font-size: 0.85rem; }
.top3-name { min-width: 200px; color: #3a3a3a; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.top3-bar-wrap { flex: 1; background: #e8f5e8; border-radius: 4px; height: 8px; overflow: hidden; }
.top3-bar { height: 100%; background: #2d7a2d; border-radius: 4px; }
.top3-pct { min-width: 42px; text-align: right; color: #2d7a2d; font-weight: 600; }
.plants-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.plant-chip { background: #e8f5e8; color: #1a5c1a; border: 1px solid #a8d8a8; border-radius: 20px; padding: 0.25rem 0.75rem; font-size: 0.8rem; font-weight: 500; }
.app-footer { text-align: center; padding: 1.5rem; color: #666; font-size: 0.82rem; margin-top: 1rem; }
"""

# ============================================================
# GRADIO INTERFACE
# ============================================================

HEADER_HTML = """
<div class="app-header">
    <span class="app-leaf-icon">🌿</span>
    <div class="app-title">PlantCare AI</div>
    <div class="app-subtitle">AI-powered plant disease detection & treatment recommendations</div>
</div>
"""

PLANTS_HTML = """
<div style="background:#fff;border-radius:12px;padding:1rem 1.25rem;border:1px solid #d4e8d4;margin-top:0.5rem;">
    <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#2d7a2d;margin-bottom:0.5rem;">Supported Plants</div>
    <div class="plants-grid">
        <span class="plant-chip">🍎 Apple</span><span class="plant-chip">🫐 Blueberry</span>
        <span class="plant-chip">🍒 Cherry</span><span class="plant-chip">🌽 Corn</span>
        <span class="plant-chip">🍇 Grape</span><span class="plant-chip">🍊 Orange</span>
        <span class="plant-chip">🍑 Peach</span><span class="plant-chip">🫑 Bell Pepper</span>
        <span class="plant-chip">🥔 Potato</span><span class="plant-chip">🍓 Strawberry</span>
        <span class="plant-chip">🫘 Soybean</span><span class="plant-chip">🎃 Squash</span>
        <span class="plant-chip">🍅 Tomato</span><span class="plant-chip">🌱 Raspberry</span>
    </div>
</div>
"""

TIPS_HTML = """
<div style="background:#fff;border-radius:12px;padding:1rem 1.25rem;border:1px solid #d4e8d4;margin-top:0.75rem;">
    <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#2d7a2d;margin-bottom:0.5rem;">📸 Tips for Best Results</div>
    <ul style="list-style:none;padding:0;margin:0;">
        <li style="font-size:0.85rem;color:#3a3a3a;padding:0.3rem 0;border-bottom:1px solid #f0f0f0;">→ Photograph a single leaf in bright natural light</li>
        <li style="font-size:0.85rem;color:#3a3a3a;padding:0.3rem 0;border-bottom:1px solid #f0f0f0;">→ Fill the frame with the leaf — avoid cluttered backgrounds</li>
        <li style="font-size:0.85rem;color:#3a3a3a;padding:0.3rem 0;border-bottom:1px solid #f0f0f0;">→ Ensure the image is sharp and in focus</li>
        <li style="font-size:0.85rem;color:#3a3a3a;padding:0.3rem 0;">→ For disease detection, photograph the most affected leaf</li>
    </ul>
</div>
"""

FOOTER_HTML = """
<div class="app-footer">
    PlantCare AI — Powered by EfficientNet-B4 trained on PlantVillage + PlantDoc datasets<br>
    <span style="color:#aaa;">For educational purposes. Always consult a qualified agronomist for critical crop decisions.</span>
</div>
"""

with gr.Blocks(css=CSS, title="PlantCare AI") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Leaf Image",
                elem_classes=["upload-section"],
                height=320,
            )
            analyse_btn = gr.Button(
                "🔍 Analyse Leaf",
                elem_classes=["analyse-btn"],
            )
            gr.HTML(PLANTS_HTML)
            gr.HTML(TIPS_HTML)

        with gr.Column(scale=1):
            result_output = gr.HTML(
                value="""
                <div style="background:#fff;border-radius:16px;padding:3rem 2rem;
                            text-align:center;border:2px dashed #d4e8d4;color:#888;font-size:0.95rem;">
                    <div style="font-size:3rem;margin-bottom:1rem;">🌱</div>
                    <div style="font-weight:600;color:#2d7a2d;margin-bottom:0.5rem;">Ready to Analyse</div>
                    <div>Upload a leaf image and click Analyse Leaf<br>to get instant disease detection and treatment advice.</div>
                </div>
                """,
                label="Analysis Result",
            )

    analyse_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=result_output,
    )

    gr.HTML(FOOTER_HTML)

demo.launch()