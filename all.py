#%%Algoritmo_de_AI_VHM
"""
Created on Thu Aug 26 11:31:03 2021

algoritmo unifcador de todo lo que se hizo

@author: Matador
"""
#%% librerias de python
import pandas as pd # manejo de dataframes
from sklearn.preprocessing import OrdinalEncoder# hacer encoder sobre las variables string
from sklearn.model_selection import train_test_split # para partir las variables
from sklearn.neighbors import KNeighborsClassifier # modelo para entrenar
import json # importar archivos json
from progress.bar import Bar #animacion de barra de carga
import numpy as np # manejo de herramientas matematicas y control de arrays
from datetime import datetime
import os

# librerias propias
from Lib.sql_operations import admin
import Lib.conect as conect
#%% objetos
#%%
#objeto para importar la informacion desde la base de datos

class subdivider():#subdivide los datos ordenados en categorias para trabajar individualmente
    
    # info son los datos completos y X son los valores a subdividir
    def __init__(self,info,col):
        """[inicializar el objeto]

        Args:
            info ([dataframe]): [base de datos modificada]
            col ([type]): [description]
        """
        self.info = info
        self.vals = split(col)
    
    def Subdivide(self,tam):
        """[subdividir los datos obtenidos por tipo de equipo]

        Returns:
            [list]: [datos segmentados]
        """
        sub = {}
        
        for i in self.vals:
            if self.info[self.info["categoria"] == i].shape[0] > tam:
                
                sub[i] = (self.info[self.info["categoria"] == i].drop(["categoria"], axis = 1))     
        
        return sub
    
class Encoder():
    
    def __init__(self,db):
        """[inicialiar el objeto]

        Args:
            db ([dataframe]): [base de datos consultada con los valores cualitativas]
        """
        self.db = db
        self.check()

    
    def check(self):
        """[estandarizar los valores de las observaciones para obtener demasiadas etiquetas y convertirlo en un problema binario]
        """
        
        y = self.db["observaciones"].to_numpy()
        
        #palabras que siguinifica que esta funcionando correctamente
        words = ["correctamente","Correctamente","correcto","buenas condiciones"]
        
        for jj,i in enumerate(y):
            for j in words:
                if i.find(j) > 0:
                    y[jj] = "ok"
                    break
                else: 
                    y[jj] = "no ok"
        
        #realizar encoder y guardar los valores
        self.enc_y = OrdinalEncoder()
        
        self.enc_y.fit(y.reshape(-1,1))
                          
    def encoder_X(self,names):
        """[convertir las variables cualitativas a cuantativas]

        Args:
            names ([list]): [nombre de la columna para aplicar encoder]
        """
        # realiza el encoder de varias columnas a la vez
    
    # diccionario que contedra los encoder
        dic = {}
        for i in names:
            
            X = np.copy(self.db[i].to_numpy().reshape(-1,1))
            
            enc = OrdinalEncoder()
            enc.fit(X)   
            dic[i] = enc
            
        return  dic
    
    def encoder_y(self):
        return self.enc_y
    
#objeto que transforma las varables de fechas a dias           
class DateTime():

    def __init__(self,db):
        """[inicializar el objeto]

        Args:
            db ([dataframe]): [datos consultados de la base de datos]
        """
        self.Xo = db
        
    def datetime2days(self):
        """[convertir las fechas de registro y de mantenimiento en dias]

        Returns:
            [datetime]: [dias entre las dos fechas]
        """
        label_tiempo = ["fecha_registro","fecha_mant"]
        
        for i in label_tiempo:
            self.Xo[i] = pd.to_datetime(self.Xo[i])
        
        self.Xo["tiempo"] = (self.Xo["fecha_mant"] - self.Xo["fecha_registro"]).apply(lambda r:r.days)
        
        self.Xo = self.Xo.drop(label_tiempo, axis = 1)
        
        return self.Xo
    
#objeto que entrena modelos
class Model():
    
    def __init__(self,model,X,y):
        """[objeto que entrena los modelos de machine learning]

        Args:
            model ([sklearn objet]): [modelo que se pretende implementar]
            X ([dataframe]): [matrix de caracteristicas]
            y ([dataframe]): [vector de etiquetas]
        """
        
        self.my_model = model
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X, y.ravel(), test_size=0.33, random_state=42)
        
    def fit(self):
        """[entrenamiento del modelo]
        """
        self.my_model.fit(self.X_train,self.y_train)
    
    def score(self):
        """[evaluar el desempeño del modelo]

        Returns:
            [float]: [desempeño del modelo]
        """
        return self.my_model.score(self.X_test,self.y_test)
    
    def predict(self,X):
        """[evaluar modelo prediciendo una muestra]

        Args:
            X ([dataframe]): [matriz de caracterisitcas]

        Returns:
            [int]: [valor predecido entre las etiquetas]
        """
        return self.my_model.predict(X)
#%% funciones
def split(X): #Obtener los valores del array
    """[adquiere los valores dentro de un array]

    Args:
        X ([list): [vector con los valores a encontrar]

    Returns:
        [list]: [valores sin repetir]
    """
    vals = []
    for i in X:
        if i not in vals:
            vals.append(i)
    return vals

def change_date(db, New_date): 
    
    """[cambia la fecha lei por la base de datos por una nueva para poder realizar la prediccion]

    Returns:
        [dataframe]: [arreglo cambiado]
    """
        
    fechas = db["fecha_registro"]
    
    new_time = datetime.strptime(New_date,"%Y-%m-%d")
    
    date_list = []
    
    for i in fechas:
        date_list.append((new_time - i).days)
    
    return date_list

class my_main():
    
    def __init__(self,cursor,connection):
        
        # f1 = open("Json/input.json")
        # self.data1 = json.load(f1)
        
        f2 = open("Json/predict.json")
        self.data2 = json.load(f2)
        self.data2["update"] = datetime.now()
        
        #cursor que realizara las operaciones diferentes a las consultas con salida de tabla
        self.c = cursor
        
        #variable para hacer las consultas de tablas
        self.db = connection
        
        # consulta que se realiza
        self.my_query = open(os.path.join('query','query.txt'),'r').read()
                
        self.columns = ["id",
                        "modelo",
                        "area",
                        "marca",
                        "nombre_sede",
                        "categoria",
                        "fecha_registro",
                        "fecha_mant",
                        "observaciones"]
        
        self.names = ["modelo",
                      "area",
                      "nombre_sede",
                      "marca"]
                
        self.train_model()
        
        if self.data2["id"] ==  "all" :
            self.predict_and_go()
        else:
            self.predict_few()

    def train_model(self):
        """[entrena el modelo]
        """
        #lectura de los datos que se encuentran en json
        self.obj_mod = {}
        
        #conectando con la base de datos
        self.Admin = admin(self.db,
                           self.c)

        #realiza la consulta
        result = conect.run_query(self.my_query,
                                  self.db)
        
        #tamaño minino de muestras
        tam = 40
        
        self.get_matrix(result,tam, training = True)
        
        columns_names = self.sub_data.keys()
        
        #definir la barra de descarga
        bar = Bar('training model:', max = len(columns_names))
        
        for i in columns_names:
            #objeto entrenador de modelo

            knn = KNeighborsClassifier(n_neighbors=3)
            self.obj_mod[i] = Model(knn,self.enc_X_data[i], self.enc_y_data[i]["label"])
            self.obj_mod[i].fit()
            bar.next()
        
        bar.finish()
             
    def predict_and_go(self):
        """[predice y envia los datos a una tabla]
        """
        # los encoders ya se encuentran construidos
        predict_data = {}
        col_names = self.enc_X_data.keys()
         
        
        bar = Bar('predicting in date {}:'.format(self.data2["new_date"]), max = len(col_names))
        
        for i in col_names:
            
            self.enc_X_data[i] = self.enc_X_data[i].drop_duplicates(subset = ["idequipos"])
            self.sub_data[i] = self.sub_data[i].drop_duplicates(subset = ["idequipos"])
            
            self.enc_X_data[i]["tiempo"] = change_date(self.sub_data[i], self.data2["new_date"])
            
            predict_data[i]  = self.obj_mod[i].predict(self.enc_X_data[i])
                        
            encoder = self.enc_y[i]
            dato = np.array(predict_data[i]).reshape(-1,1)        
            predict_data[i] = encoder.inverse_transform(dato)
            
            bar.next()
            
        bar.finish()
        
        #generar la tabla de prediciones

        new_table = []

        for i in col_names:
            for j in range(len(self.sub_data[i])):
                
                new_table.append((predict_data[i][j][0],
                                  self.data2["new_date"],
                                  self.data2["update"],
                                  int(self.sub_data[i]["idequipos"].to_numpy()[j])))
                
        table_name = "predicciones"
        variable_name = ["predición","fecha_p","fecha_a","equipos_idequipos"]
        types = ["VARCHAR (255)", "DATE","DATE","INT"]

        exists = self.Admin.create_table(table_name, variable_name, types)  
        
        #inserta todos los datos
        if exists: 
            self.Admin.add_foreigh_key("equipos",table_name)
            
            for i in new_table:
                
                self.Admin.add_info(table_name,
                                    variable_name,
                                    i,
                                    exists,
                                    types,
                                    check_pos = 0)
        
        else:
            
            bar2 = Bar('update information: ', max = len(new_table))
            
            query = "SELECT * FROM {}".format(table_name)
            
            table = conect.run_query(query,self.db)
            
            id_table = table["equipos_idequipos"].to_numpy()
                        
            for i in new_table:
                
                if i[3] in id_table:
                    exists = False
                else:
                    exists = True
            
                # este vector es para tomar la decision entre insertar la informacion o actualizarla    
                self.Admin.add_info(table_name,
                                    variable_name,
                                    i,
                                    exists,
                                    types,
                                    check_pos = 3)
                bar2.next()
        
            bar2.finish()

        "insertando o actualizando los datos en la tabla"

    def get_matrix(self, result, num, training):
        # carga la informacion
        """[summary]

        Args:
            result ([lista de tuplas]): [resultado de la consulta]
            num    ([int]): [numero minimo de consultas]
            training ([bool]): [estado]
        """
        if training: 
            # variables donde se encontrara el encoder
            self.enc_X = {}
            self.enc_y = {}
        
        # los valores que se les realiazar el encoder
        self.enc_X_data = {}
        self.enc_y_data = {}
        
        self.days_data = {}
        
        #subdivimos la muestra
        obj_sub = subdivider(result, result["categoria"])
        self.sub_data = obj_sub.Subdivide(num)
        
        #columna a realizar encoder 
        columns_names = self.sub_data.keys()
                
        for j,i in enumerate(columns_names):
                        
            #objeto tiempo
            obj_time = DateTime(self.sub_data[i])
            self.days_data[i] = obj_time.datetime2days()

            #objeto de encoder
            obj_enc = Encoder(self.days_data[i])
            
            # si se esta llevando a cabo el entrenamiento
            if training:
                
                self.enc_X[i] = obj_enc.encoder_X(self.names)
                self.enc_y[i] = obj_enc.encoder_y()
            
            aux = self.enc_y[i].transform(np.array(self.days_data[i]["observaciones"]).reshape(-1,1)) 
            self.enc_y_data[i] = pd.DataFrame(aux,columns = ["label"])
            
            
            dic = {}
            dic["idequipos"] = self.days_data[i]["idequipos"]
            
            for i_names in self.names:
                    
                dic[i_names] = self.enc_X[i][i_names].transform(np.array(self.days_data[i][i_names]).reshape(-1,1))
                dic[i_names] = dic[i_names].ravel()
                
            dic["tiempo"] = self.days_data[i]["tiempo"]    
            
            self.enc_X_data[i] = pd.DataFrame(dic)
            
            l = len(self.enc_X_data[i])
            rango = list(range(l))
            index = pd.Index(rango)
            
            self.enc_X_data[i] = self.enc_X_data[i].set_index([index])            
        
    def predict_few(self):
        """[predice los id indicados en predict.json]
        """
        word = "equi.idequipos = {}"
        self.my_query += " AND ("
        
        for i in self.data2["id"]:
            
            word = " equi.idequipos = {} OR".format(i)
            self.my_query +=  word
        
        self.my_query = self.my_query[:len(self.my_query) - 2]
        self.my_query += " )"
        
        result = conect.run_query(self.my_query, self.db)
        
        self.get_matrix(result,num = 1, training = False)
        
        self.out = self.enc_X_data
        
        self.predict_and_go()
    
    def r(self):
        return self.out

def open_shh(ssh,database):
    
        
    tunel = conect.open_ssh_tunnel(ssh['host'],
                                   ssh['user'],
                                   ssh['passwd'],
                                   verbose = False)
    
    connection = conect.mysql_connect(database['user'],
                                      database['passwd'],
                                      database['name'],
                                      tunel)
    
    cursor = connection.cursor()
    
    return cursor, connection

if __name__ == "__main__":
    
    ssh = {}
    database = {}
    
    ssh['host'] =     '107.180.51.23'
    ssh['user'] =     'zda7zbr6kwzz'
    ssh['passwd'] =   'Agosto1994.'
    
    # database information
    
    database['user'] =   'axel_garcia'
    database['passwd'] = '4x3lg4rc14'
    database['name'] =   'axel_test'
    localhost =          '127.0.0.1'
    
    # link progress
    
    cursor, connection = open_shh(ssh,database)
    
    obj = my_main(cursor,connection)