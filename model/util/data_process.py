import json

def load_data(sql_path,table_path):
    sql_data = []
    table_data = {}
    # read sql data
    with open(sql_path) as inf:
        for line in inf:
            sql = json.loads(line.strip())
            sql_data.append(sql)
    inf.close()

    # read table data

    with open(table_path) as inf:
        for line in inf:
            table = json.loads(line.strip())
            table_data[table[u'id']] = table

    return sql_data,table_data

