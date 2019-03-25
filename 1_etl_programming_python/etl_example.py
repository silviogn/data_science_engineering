import petl as etl
import psycopg2 as pg 
import sys

dbCnxns = {
    'olap':'dbname=sd_copy user=postgres password=****** host=127.0.0.1',
    'oltp':'dbname=sd user=postgres password=****** host=127.0.0.1'
    }

'''Create the connections'''
sourceConnection = pg.connect(dbCnxns['oltp'])
targetConnection = pg.connect(dbCnxns['olap'])

'''Create the cursors'''
sourceCursor = sourceConnection.cursor()
targetCursor = targetConnection.cursor()

sourceCursor.execute("""select table_name from information_schema.columns 
where table_name in ('ds_0008','ds_0010') group by 1""")

sourceTables = sourceCursor.fetchall()

# Interate over the tables to copy it.
for table in sourceTables:
    print('Processing table {}'.format(table[0]))
    targetCursor.execute("drop table if exists {} ".format(table[0]))
    sourceDs = etl.fromdb(sourceConnection, "select * from {}".format(table[0]))
    etl.todb(sourceDs, targetConnection, table[0], create=True, sample=10000)

print("End.")