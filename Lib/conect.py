#%% libraries

import pandas as pd # manage dataframe
import pymysql      # conect my database
import logging      #
import sshtunnel    # conect to the database through a ssh tunel
from sshtunnel import SSHTunnelForwarder #function for activate ssh tunnel

#%% functions

def open_ssh_tunnel(ssh_host, ssh_username, ssh_password,verbose):
    
    if verbose:
        sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG
    
    global tunnel
    
    tunnel = SSHTunnelForwarder(
        (ssh_host, 22),
        ssh_username = ssh_username,
        ssh_password = ssh_password,
        remote_bind_address = ('127.0.0.1', 3306)
    )
    
    tunnel.start()
    
    return tunnel

def mysql_connect(database_username, database_password, database_name,tunel):
    
    global connection
    
    connection = pymysql.connect(
        host='127.0.0.1',
        user=database_username,
        passwd=database_password,
        db=database_name,
        port=tunnel.local_bind_port
    )
    
    return connection

def run_query(sql,connection):

    return pd.read_sql_query(sql, connection)


#%% code

if __name__ == '__main__':

    ssh_host = '107.180.51.23'
    ssh_username = 'zda7zbr6kwzz'
    ssh_password = 'Agosto1994.'
    
    database_username = 'axel_garcia'
    database_password = '4x3lg4rc14'
    database_name = 'axel_test'
    localhost = '127.0.0.1'
    
    
    open_ssh_tunnel()
    mysql_connect()
    df = run_query("SELECT * FROM equipos")
    df.head()
    
    