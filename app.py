#librerias
import streamlit as st
import pandas as pd
from pycaret.classification import setup, load_model,predict_model
from streamlit_extras.let_it_rain import rain 
from PIL import Image
import plotly.graph_objects as go
import joblib


st.set_page_config(page_title="CoraTest",page_icon="❤️",layout="wide")
#extract archivos
model = joblib.load('Models\ModelLastNB.pkl')
#model = joblib.load('nb_model.pkl') # Carga del modelo.
data=pd.read_csv('CSV\heart_2022_no_nans.csv')
#df_unido = pd.read_excel('CSV\causes_of_death_16_17.xlsx')
df_unido=pd.read_csv('CSV\causes_of_death_16_17.csv')

def predictor(model,df):
    predictions_data = predict_model(estimator = model, data = df) 
    return predictions_data['prediction_label'][0]
#funcion para clasificar
def classify(num):
    if num==0:
        example_no()
        return "No tiene riesgo considerable de sufrir enfermedades cardíacas😊"
    else:
        return "SÍ tiene un CONSIDERABLE riesgo de sufrir enfermedades cardíacas , Es recomendable visitar un Médico" 
def example_no():
        rain(
        emoji="❤️",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",)
#Conversion de los datos 
def conversion(df):
    #conversiones manuales (。_。)
    df['Sexo']=df['Sexo'].map({'Mujer':0,'Hombre':1}).astype(float)
    df['CategoríaEdad'] = df['CategoríaEdad'].map({'Edad 18 a 24': 0,'Edad 25 a 29': 1,'Edad 30 a 34': 2,'Edad 35 a 39': 3,'Edad 40 a 44': 4,'Edad 45 a 49': 5,'Edad 50 a 54': 6,
                            'Edad 55 a 59': 7,'Edad 60 a 64': 8,'Edad 65 a 69': 9,'Edad 70 a 74': 10,'Edad 75 a 79': 11,'Edad 80 o más': 12}).astype(float)
    df['CategoríaRazaEtnia'] = df['CategoríaRazaEtnia'].map({'Blanco/a , No Hispano/a': 0,'Hispano/a': 1,'Negro/a , No Hispano/a': 2,'Otra raza, No Hispano/a': 3,
                                                                            'Multirracial, No Hispano/a': 4}).astype(float)
    df['EstadoFumador'] = df['EstadoFumador'].map({'Ex fumador/a': 0,'Nunca ha fumado': 1,'Fumador/a actual - fuma todos los días': 2,
                                                            'Fumador/a actual - fuma algunos días': 3}).astype(float)
    df['UsoCigarrilloElectrónico']=df['UsoCigarrilloElectrónico'].map({'Nunca he usado cigarrillos electrónicos en mi vida':0,'Los uso algunos días':1,
                                                                                'En absoluto (en este momento)':2,'Los uso todos los días':3 }).astype(float)
    df['Tiene Diabetes']=df['Tiene Diabetes'].map({'No': 0,'Sí': 1,'Sí, pero solo durante el embarazo (mujer)': 2,
                                                            'No, pre-diabetes o borderline diabetes': 3}).astype(float)
    df['BMI(Índice de Masa Corporal)']=df['BMI(Índice de Masa Corporal)'].astype(float)
    df['DíasSaludMental']=df['DíasSaludMental'].astype(float)
    columnas = ['BebedoresAlcohol','Actividades Fisicas','PruebaVIH']
    for columna in columnas:
        df[columna] = df[columna].map({'No': 0, 'Sí': 1}).astype(float)
    return df

st.title('Riesgo de Enfermedad Cardíaca')  
tab1, tab2 = st.tabs(["INFO📉", "TEST🩺"])
tab1.title("Tendencias preocupantes. Enfermedades cardíacas en Estados Unidos")
tab2.title("CORATest❤️")

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('##')
        st.write("""En las últimas décadas, el aumento de enfermedades cardíacas en Estados Unidos ha sido alarmante. 
Factores como el sedentarismo, la dieta poco saludable, el estrés y el tabaquismo se han convertido en pilares del incremento de este riesgo.                                                                                                                                        
El mapa de la derecha muestra la distribución de la media del Índice de Masa Corporal (BMI) por estado en los Estados Unidos, 
elaborado con datos del 2022, recopilados por el CDC. 
""")
        st.markdown('**AQUÍ TIENES ALGUNOS DATOS**')
        st.write("""Un gráfico circular muestra el porcentaje de las tres principales causas de muerte por enfermedades en EEUU en 2017, 
siendo las enfermedades cardíacas la causa predominante. 
Por último, se ilustra la evolución de enfermedades cardíacas específicas(accidente cerebrovascular,angina,ataque al corazón)entre 2015 y 2017.
¡Explora estos datos esenciales para comprender mejor la situación de la salud cardiovascular en EEUU! 
        """)
        st.subheader("""Prevención cardíaca: Consejos clave para un corazón saludable""") 
        st.write("""Según referencias del CDC y la Asociación Americana del Corazón, adoptar un estilo de vida activo, mantener una dieta equilibrada y reducir el consumo de tabaco son medidas esenciales para prevenir estas enfermedades. ¡Tu bienestar es lo primero!""")
        
        st.markdown('##')
        
        image = Image.open("IMG\zules.png")
        st.image(image)
       
    with col2:
        data.drop_duplicates(inplace=True)
        Estados = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
            'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
            'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
            'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
            'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
            'New Jersey', 'New Mexico', 'New York', 'North Carolina',
            'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
            'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
            'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
            'West Virginia', 'Wisconsin', 'Wyoming', 'Guam', 'Puerto Rico',
            'Virgin Islands']
        Abreviacion = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
        StateShort = dict(zip(Estados,Abreviacion))
        data['StateShort'] = data['State'].map(StateShort)
        BMI_State=data[['BMI','StateShort']]
        group=BMI_State.groupby('StateShort')['BMI'].mean()
        estados = data['StateShort'].unique()
        # los valores correspondientes
        BMI_State=data[['BMI','StateShort']]
        valores=BMI_State.groupby('StateShort')['BMI'].mean()
        fig = go.Figure(data=go.Choropleth(
                        locations=estados,
                        z=valores,
                        locationmode='USA-states',
                        colorscale='Blues',
                        colorbar_title="Media BMI",
                        marker_line_color='rgba(0,0,0,0)',  # Bordes transparentes
                        marker_line_width=0,  # Ancho de los bordes a cero para que sea transparente
                        hoverinfo='location+z',  # Información sobre herramientas al pasar el ratón
                        zmin=27,  # Valor mínimo para el rango de color
                        zmax=30,  # Valor máximo para el rango de color
        ))
        # Configuración adicional del diseño
        fig.update_layout(
                        title_text='Media de BMI (Índice de Masa Corporal) por estado',
                        geo_scope='usa',
                        template='plotly_dark',
                        geo=dict(bgcolor='rgba(0,0,0,0)'),  # Fondo del mapa transparente
                        paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente del área fuera del gráfico
                        plot_bgcolor='rgba(0,0,0,0)',    # Fondo transparente del área del gráfico
        )
        # Mostrar el gráfico en la aplicación Streamlit
        st.plotly_chart(fig)  
        image = Image.open("IMG\output.png")
        st.image(image)
with tab2:
    st.write("""¡Bienvenido al CoraTest, tu herramienta para una rápida estimación del riesgo cardíaco!
                Desarrollado por estudiantes de análisis de datos comprometidos con la salud pública.""")
    st.write("""Basado en un modelo de regresión logística con una precisión del 96%, este test es una herramienta de evaluación rápida. Utiliza datos del CDC para predecir el riesgo cardíaco.""")
    st.write("""¿Listo para conocer más sobre tu salud cardíaca en segundos?""") 
    col3, col4 ,col5 ,col6 ,col7 = st.columns(5)
    with col3:
        st.subheader('Características Generales')
        Sexo=st.selectbox('Sexo Biológico',options=['Mujer','Hombre'])
        CategoríaEdad=st.selectbox('Categoría de Edad', options=['Edad 18 a 24','Edad 25 a 29','Edad 30 a 34','Edad 35 a 39','Edad 40 a 44','Edad 45 a 49','Edad 50 a 54',
                            'Edad 55 a 59','Edad 60 a 64','Edad 65 a 69','Edad 70 a 74','Edad 75 a 79','Edad 80 o más'])
        CategoríaRazaEtnia=st.selectbox('Raza/Etnia',options=['Blanco/a , No Hispano/a','Hispano/a','Negro/a , No Hispano/a','Otra raza, No Hispano/a',
                                                               'Multirracial, No Hispano/a'])
        st.subheader("¿ Ha realizado el siguiente TEST ?")
        PruebaVIH=st.selectbox('¿Test VIH?',options=['Sí','No'])
        
    with col4:
        st.subheader("Factores de riesgo en tu estilo de vida")
        BMI=st.slider('BMI(Índice de Masa Corporal)',min_value=12, max_value=43, value=1)
        UsoCigarrilloElectrónico=st.selectbox('Uso de Cigarrillo Electrónico',options=['Nunca he usado cigarrillos electrónicos en mi vida',
                                               'Los uso algunos días','En absoluto (en este momento)','Los uso todos los días'])
        EstadoFumador=st.selectbox('Tipo de Fumador',options=['Ex fumador/a','Nunca ha fumado','Fumador/a actual - fuma todos los días',
                                               'Fumador/a actual - fuma algunos días'])
        BebedoresAlcohol=st.selectbox('¿Toma Alcohol?',options=['Sí','No'])
        ActividadesFisicas=st.selectbox('¿Realiza Actividades Físicas?',options=['Sí','No'])
    with col5:
        st.subheader('Condición médica preexistente')
        DíasSaludMental=st.slider('¿Cuantos dias al mes diria que no se siente del todo bien ?',min_value=0, max_value=30, value=1)
        TieneDiabetes=st.selectbox('¿Tiene Diabetes?',options=['No','Sí','Sí, pero solo durante el embarazo (mujer)',
                                               'No, pre-diabetes o borderline diabetes'])
        input_data = pd.DataFrame([[Sexo,CategoríaEdad,CategoríaRazaEtnia,EstadoFumador,UsoCigarrilloElectrónico,BebedoresAlcohol,BMI,
                                    ActividadesFisicas,DíasSaludMental,TieneDiabetes, PruebaVIH
                                    ]],
                                    columns=['Sexo','CategoríaEdad','CategoríaRazaEtnia','EstadoFumador','UsoCigarrilloElectrónico','BebedoresAlcohol','BMI(Índice de Masa Corporal)',
                                    'Actividades Fisicas','DíasSaludMental','Tiene Diabetes','PruebaVIH'])
        
        st.markdown('##')
        st.markdown('##')
        st.markdown('##')
        
        if st.button('Veamos el Resultado'):
            #st.success(classify(predictor(model,conversion(input_data))))
            predicciones = model.predict(conversion(input_data))
            st.success(classify(predicciones))
    with col6:
        st.markdown('##')
        image = Image.open("IMG\Ricardio.png")
        st.image(image, caption='Ricardio')
    with col7:  
        st.markdown("# ⚠️AUNQUE NO REEMPLAZA LA CONSULTA MÉDICA⚠️")