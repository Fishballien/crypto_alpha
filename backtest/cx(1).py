import pymysql

实盘

mysqldb_conn = pymysql.connect(host='localhost', port=3306, user='yuyang', passwd='helloworld', db='policycurves',charset='utf8')
cursor = mysqldb_conn.cursor()

for index, row in result_df.iterrows():

	# print '===================================='
	# print nowtime

	 for comm in comms:
		ps = row[comm]

		sql = "insert into policycurves.position set acname='%s', name='%s', product='%s', ps=%s, if_final=%s on duplicate key " \
			"update ps = %s, if_final=%s;" % (acname, 'strategy1', comm, ps, True, ps, True)

		cursor.execute(sql)
		mysqldb_conn.commit()


cursor.close()
mysqldb_conn.close()



回测


mysqldb_conn = pymysql.connect(host='115.159.152.138', port=3306, user='pyprogram', passwd='K@ra0Key', db='btcfuture',charset='utf8mb4')
cursor = mysqldb_conn.cursor()

eu.multiprocess(func=writetosql, paras=comms, n_processes=2, fps_df=fps_df, acname=acname, cursor=cursor, conn=mysqldb_conn)

cursor.close()
mysqldb_conn.close()



def writetosql(comm, fps_df, acname, cursor, conn):

    try:
        sr_ps = fps_df[comm]
        idx = (sr_ps != sr_ps.shift(1)).values
        sr_ps = sr_ps[idx]

        for i in range(sr_ps.shape[0]):
            sql = "insert into btcfuture.python_trade_backtest (stockdate, acname, symbol, position) values" \
                  "('%s', '%s', '%s', %s);\n" % (sr_ps.index[i], acname, comm, sr_ps[i])

            cursor.execute(sql)
            conn.commit()

        print 'insert the ps of %s in %s '% (comm, acname)

    except Exception, exception:

        traceback.print_exc()




		
tmp_tdfavg = tdf[comms]
tdfavg = tmp_tdfavg.resample('D').last()
tdfavg.index = tdfavg.index + pd.DateOffset(minutes=1439)
tdfavg = tdfavg.rolling(window=20).mean()
tdfavg = 1000.0/tdfavg
tdfavg = tdfavg.shift()
tdfavg = tdfavg.fillna(0)
tdfavg = tdfavg.reset_index()
tmp = pd.DataFrame({'index': tdf.index})
tdfavg = tdfavg.merge(right=tmp, how='right', on='index')
tdfavg = tdfavg.fillna(method='ffill')
tdfavg = tdfavg.set_index(['index'])