# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 09:13:10 2021

@author: Matador
"""

class admin():    
    #incializar variables
    def __init__(self,db,c):   
        
        """[inicializacion de objeto que se conecta con la base de datos]

        Args:
            user ([string]): [usuario registrado en la base de datos]
            passwd ([string]): [contraseña relacionada con el usuario]
            name_database ([string]): [nombre de la base de datos]
            host ([string]): [puerto al que se conecta]
        """
        self.c = c
        self.db = db
        
    #conectar con la base de datos
                    
    def create_table(self,table_name,variables_names,variables_types):
        """[crea una tabla en la base de datos]

        Args:
            table_name ([string]): [nombre de la tabla a crear]
            variables_names ([list]): [nombre de las variables de la tabla]
            variables_types ([list]): [nombre de los tipos de variables]

        Returns:
            [boolean]: [un estado si la tabla ya existia]
        """
        
        
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
        """[revisa si la tabla existe en la base de datos]

        Args:
            name ([string]): [nombre de la tabla a consultar]

        Returns:
            [boolean]: [estado de la consulta]
        """
        
        self.c.execute("SHOW TABLES")
        
        tables_names = [i[0] for i in self.c]
        
        if name not in tables_names:
            return True
        else:
            return False
        
    def add_foreigh_key(self,tablep,tables):
        """[agrega una llave foranea con respecto a las id de las tablas

        Args:
            tablep ([string]): [nombre de la tabla principal]
            tables ([type]): [nombre  de la tabla secundaria]
        """
        
        query = "ALTER TABLE " + tables
         
        ID0 = "(" + tablep + "_id" + tablep + ")" 
        ID1 = "(id" + tablep+ ")"
         
        query += " ADD FOREIGN KEY " + ID0 + " REFERENCES " + tablep + ID1 
         
        self.c.execute(query)
         
    def drop_table(self,table_name):
        """[elimina una tabla]

        Args:
            table_name ([string]): [nombre de la tabla a eliminar]
        """
        
        query = "DROP TABLE IF EXISTS {}".format(table_name)
        self.c.execute(query)
        
    def drop_column(self,table_name,column_name):
        """[elimina una columna de una tabla]

        Args:
            table_name ([string]): [nombre de la tabla]
            column_name ([string]): [nombre de la column]
        """
        query = "ALTER TABLE {} DROP COLUMN {}".format(table_name, column_name)
        
        self.c.execute(query)
        
        print("column has been deleted")
        
    def add_column(self,table_name,column_name,types):
        """[agrega columna a una tabla]

        Args:
            table_name ([string]): [nombre de la tabla]
            column_name ([string]): [nombre de la nueva columna]
            types ([string]): [nombre del tipo de dato de la columna]
        """
        
        query = "ALTER TABLE {} ADD {} {}".format(table_name,column_name,types)
        self.c.execute(query)
        
    def is_empty(self,table_name):
        """[revisa si una tabla se encuentra vacia]

        Args:
            table_name ([string]): [nombre de la columna]

        Returns:
            [int]: [numero de registro dentro de la tabla]
        """
        
        query = "SELECT * FROM {}".format(table_name)
        
        self.c.execute(query)
        self.c.fetchall()
        
        return self.c.rowcount
    
    
    def add_info(self,table_name,columns_name,vals,value,types,check_pos):
        """[agrega informacion dentro de la tabla]

        Args:
            table_name ([string]): [nombre de la tabla]
            columns_name ([list]]): [nombre de la columna a modificar o insertar]
            vals ([list]): [valores a ingresar]
            value ([boolean]): [revisar si la tabla apenas se creo o ya lo habia hecho]
        """
        
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
            
            val = ()
            for i,j in zip(vals,types):
                
                if j != "INT":
                    val += (i,)
                else:
                    val += (int(i),)
        
            self.c.execute(query,val)
            
            self.db.commit()
            
        else:
            check_info = [columns_name[check_pos],vals[check_pos]]
            
            for col,val in zip(columns_name,vals):
                
                update_info = [col,val]
                self.update(table_name,update_info,check_info)
            
    def update(self,table_name, update_info, check_info):#funcion que actualiza los valores
        """[actualiza los datos de la tabla]

        Args:
            table_name ([string]): [nombre de la tabla]
            update_info ([list]): [informacion para actualizar]
            check_info ([list]): [informacion de condicion donde actualizar los datos]
        """
        if(isinstance(update_info[1], str)):
        
            query = "UPDATE {} SET {} = '{}' WHERE {} = {}".format(table_name,update_info[0],update_info[1],check_info[0],check_info[1])
            
            self.c = self.db.cursor()
            self.c.execute(query)
            self.db.commit()
