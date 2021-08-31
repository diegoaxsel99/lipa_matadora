#Algoritmo_de_AI_VHM
"""
Created on Thu Aug 26 11:31:03 2021

algoritmo unifcador de todo lo que se hizo

@author: Matador
"""
#%% librerias
import mysql.connector as sql # conectar a base de datos
import pandas as pd # manejo de dataframes
from sklearn.preprocessing import OrdinalEncoder# hacer encoder sobre las variables string
from sklearn.model_selection import train_test_split # para partir las variables
from sklearn.neighbors import KNeighborsClassifier # modelo para entrenar
#%% funciones
def split(X): #Obtener los valores del array

    vals = []
    for i in X:
        if i not in vals:
            vals.append(i)
    return vals
#%% objetos

#objeto para importar la informacion desde la base de datos
class admin():    
    #incializar variables
    def __init__(self,user,passwd,name_database,host):        
        self.user = user
        self.passwd = passwd
        self.name_database = name_database
        self.host = host
        
    #conectar con la base de datos
    
    def conect(self): # conectarse con la base de datos    
        self.db = sql.connect(
                user = self.user,
                passwd = self.passwd,
                database = self.name_database,
                host = self.host
                )
        return self.db
    
   # realiza la consulta 
    def query(self,query):
        
        self.c = self.db.cursor()
        self.c.execute(query)
        
        return self.c.fetchall()
    
    def create_table(self,table_name,variables_names,variables_types):
        
        if(self.table_exist(table_name)  == True):
            
            my_query = "CREATE TABLE "
            my_query += table_name
            my_query += " (id{} INT AUTO_INCREMENT PRIMARY KEY,".format(table_name)
                        
            for i,j in zip(variables_names, variables_types):
                
                name = " " + i +" "
                Type = j  +" NOT NULL "+ ","
                
                my_query = my_query + name + Type
            
            my_query = my_query[:len(my_query) - 1]
            
            my_query += ")"
            
            self.c.execute(my_query)
            
            print("table has been created")
            return True
            
        else:
            
            print ("table already exist")
            return False
                
    def table_exist(self,name):
        
        self.c.execute("SHOW TABLES")
        
        tables_names = [i[0] for i in self.c]
        
        if name not in tables_names:
            return True
        else:
            return False
        
    def add_foreigh_key(self,tablep,tables):
         query = "ALTER TABLE " + tables
         ID0 = "(" + tablep + "_id" + tablep + ")" 
         ID1 = "(id" + tablep+ ")"
         
         query += " ADD FOREIGN KEY " + ID0 + " REFERENCES " + tablep + ID1 
        
         print(query)
        
         self.c.execute(query)
         
    def drop_table(self,table_name):
        
        query = "DROP TABLE IF EXISTS {}".format(table_name)
        self.c.execute(query)
        
    def drop_column(self,table_name,column_name):
        
        query = "ALTER TABLE {} DROP COLUMN {}".format(table_name, column_name)
        
        self.c.execute(query)
        
        print("column has been deleted")
        
    def add_column(self,table_name,column_name,types):
        
        query = "ALTER TABLE {} ADD {} {}".format(table_name,column_name,types)
        self.c.execute(query)
        
    def is_empty(self,table_name,val):
        
        query = "SELECT * FROM {}".format(table_name)
        
        self.c.execute(query)
        self.c.fetchall()
        
        return self.c.rowcount
    
    
    def add_info(self,table_name,columns_name,vals,value):
                
        if value ==   True:
            
            query = "INSERT INTO {} ".format(table_name)
            
            f = "("
            s = "("
            for i in columns_name:
                f += i + ", "
                s += "%s, "
                
            f = f[:len(f) - 2]
            f += ")"
            
            s = s[:len(s) - 2]
            s += ")"
            
            query += f + " VALUES " + s
            
            val = (vals[0],vals[1],int(vals[2]))
            self.c.execute(query,val)
            
            self.db.commit()
            
        else:
            
            check_info = [columns_name[2],vals[2]]
            
            for col,val in zip(columns_name,vals):
                
                update_info = [col,val]
                self.update(table_name,update_info,check_info)
        
                
    def check_query(self,table_name,column_name,val):
        
        query = "SELECT * FROM {} WHERE {}.{} = {}".format(table_name, table_name, column_name,val)
        
        self.c = self.db.cursor(buffered = True)
        
        
        self.c.execute(query)
        
        return self.c.rowcount
    
    def update(self,table_name, update_info, check_info):#funcion que actualiza los valores
        
        print("entro")
        if(isinstance(update_info[1], str)):
        
            query = "UPDATE {} SET {} = '{}' WHERE {} = {}".format(table_name,update_info[0],update_info[1],check_info[0],check_info[1])
            
            self.c = self.db.cursor(buffered = True)
            self.c.execute(query)
            self.db.commit()

#organizador con respecto a lo enviado
class organizer():
    
    def __init__(self,result,columns):
        
        self.messy = result
        self.columns = columns
    
    def sort(self):
        
        #crea un dicticcionario
        sort = {}
        
        #llenamos de listas
        for i in self.columns:
            sort[i] = []
        
        #ordenamos para luego generar un dataframe
        for i in self.messy:
            for jj,j in enumerate(self.columns):
                sort[j].append(i[jj])
        
        return pd.DataFrame(sort)

class subdivider():#subdivide los datos ordenados en categorias para trabajar individualmente
    
    # info son los datos completos y X son los valores a subdividir
    def __init__(self,info,col):
        self.info = info
        self.vals = split(col)
    
    def Subdivide(self):
        
        sub = {}
        
        for i in self.vals:
            if self.info[self.info["categoria"] == i].shape[0] > 40:
                
                sub[i] = (self.info[self.info["categoria"] == i].drop(["categoria"], axis = 1))     
        
        return sub
    
class Encoder():
    
    def __init__(self,db):
        self.db = db
        self.check()

    
    def check(self):
        
        self.y = self.db["observaciones"].to_numpy()
        
        #palabras que siguinifica que esta funcionando correctamente
        words = ["correctamente","Correctamente","correcto","buenas condiciones"]
        
        for jj,i in enumerate(self.y):
            for j in words:
                if i.find(j) > 0:
                    self.y[jj] = "ok"
                    break
                else: 
                    self.y[jj] = "no ok"
        
        #realizar encoder y guardar los valores
        enc = OrdinalEncoder()
        self.y = {"enc":enc, 
                  "values":enc.fit_transform(self.y.reshape(-1,1))}
                              
        
    def encoder(self,names):
        # realiza el encoder de varias columnas a la vez
    
    # diccionario que contedra los encoder
        dic = {}
        for i in names:
            enc = OrdinalEncoder()
            self.db[i] = enc.fit_transform(self.db[i].to_numpy().reshape(-1,1))    
            dic[i] = enc
        
        #metemos todo en un diccionario
        data_enc = {"enc":dic,
                   "values":self.db.drop(["observaciones"], axis = 1)}    
            
        return  {"X": data_enc, "y": self.y}
#objeto que transforma las varables de fechas a dias           
class DateTime():

    def __init__(self,db):
        self.Xo = db
        
    def datetime2days(self):
        
        label_tiempo = ["fecha_registro","fecha_mant"]
        
        for i in label_tiempo:
            self.Xo[i] = pd.to_datetime(self.Xo[i])
        
        self.Xo["tiempo"] = (self.Xo["fecha_mant"] - self.Xo["fecha_registro"]).apply(lambda r:r.days)
        
        self.Xo = self.Xo.drop(label_tiempo, axis = 1)
        
        return self.Xo
    
#objeto que entrena modelos
class Model():
    
    def __init__(self,model,X,y):
        
        self.my_model = model
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( X, y.ravel(), test_size=0.33, random_state=42)
        
    def fit(self):
        self.my_model.fit(self.X_train,self.y_train)
    
    def score(self):
        return self.my_model.score(self.X_test,self.y_test)
    
    def predict(self,X):
        return self.my_model.predict(X)
 
#%% Funciones

def change_date(db,New_date): 
    
    list_date = list()
    list_date.append(New_date)
    
    amoung = len(db["fecha_mant"])
    
    list_date = amoung*list_date
    
    db["fecha_mant"] = list_date
    
    obj_change_time = DateTime(db)
    db = obj_change_time.datetime2days()
    
    return db["tiempo"]
        
#%%       
user = "axsel"
passwd = "17060327A"
database = "axsel_test"
host = "localhost"

#conectando con la base de datos
Admin = admin(user,passwd ,database,host)
Admin.conect()

# consulta que se realiza
my_query = """SELECT 

equi.idequipos,
modelo.modelo,
areas.area,
marca.marca,
sede.nombre_sede,
cat_equi.categoria,
equi.fecha_registro, 
mp.fecha_mant, 
mp.observaciones

FROM 
equipos AS equi, 
mant_prevent AS mp,
modelo,
areas_servicios AS areas,
categoria_equipos AS cat_equi,
sede_empresa AS sede,
marca

WHERE 
equi.idequipos = mp.equipos_idequipos 
AND 
equi.modelo_idmodelo = modelo.idmodelo
AND
equi.areas_servicios_idareas_servicios = areas.idareas_servicios
AND
equi.categoria_equipos_idcategoria_equipos = cat_equi.idcategoria_equipos
AND
equi.sede_empresa_idsede_empresa = sede.idsede_empresa
AND
modelo.marca_idmarca = marca.idmarca"""

#realiza la consulta
result = Admin.query(my_query)

#ordenar los datos obtenidos
columns = ["id",
           "modelo",
           "area",
           "marca",
           "nombre_sede",
           "categoria",
           "fecha_registro",
           "fecha_mant",
           "observaciones"]

#definimos objeto para ordenar
obj_org = organizer(result,columns)
sort_data = obj_org.sort()

#subdivimos la muestra
obj_sub = subdivider(sort_data, sort_data["categoria"])
sub_data = obj_sub.Subdivide()

#correcion de las fechas a dias,encoder sobre las variables, entrenamiento de modelo seleccionado
days_data = {}
enc_data = {}
predict_data = {}

#columna a realizar encoder 
names = ["modelo","area","nombre_sede","marca"]
scores = []

new_date = "2023-02-01"

columns_names = sub_data.keys()

for i in columns_names:
    
    # objeto de tiempo
    obj_time = DateTime(sub_data[i])
    days_data[i] = obj_time.datetime2days()
    
    #objeto de encoder
    obj_enc = Encoder(days_data[i])
    enc_data[i] = Encoder.encoder(obj_enc,names)
    
    #objeto entrenador de modelo
    
    knn = KNeighborsClassifier(n_neighbors=3)
    obj_mod = Model(knn,enc_data[i]["X"]["values"], enc_data[i]["y"]["values"])
    obj_mod.fit()
    
    enc_data[i]["X"]["values"]["tiempo"] = change_date(sub_data[i], new_date)
    
    predict_data[i]  = obj_mod.predict(enc_data[i]["X"]["values"])
    predict_data[i] = enc_data[i]["y"]["enc"].inverse_transform(predict_data[i].reshape(-1,1))
    
    scores.append(obj_mod.score())

#generar la tabla de prediciones

new_table = []

for i in columns_names:
    for j in range(len(sub_data[i])):
        
        new_table.append((predict_data[i][j][0],new_date,int(sub_data[i]["id"].to_numpy()[j])))
        
#%%

# creando una tabla en la base de datos
table_name = "predicciones"
variable_name = ["predición","fecha","equipos_idequipos"]
types = ["VARCHAR (255)", "DATE","INT"]

exists = Admin.create_table(table_name, variable_name, types)  
if(exists):
    Admin.add_foreigh_key("equipos",table_name)

"insertando o actualizando los datos en la tabla"

for i in new_table:
    
    # este vector es para tomar la decision entre insertar la informacion o actualizarla    
    Admin.add_info(table_name,variable_name,i,exists)
#%%

