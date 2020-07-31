#!/usr/local/bin/python3.8
#coding: utf-8
import pandas as pd
import numpy as np
from datetime import timedelta
import gc


class Handle():
    def __init__(self):
        #读取2018/4/1 ~ 2018/4/15 的数据，共15天
        jd_consumer = pd.read_csv('../data/京东消费者分析数据2018-04.csv')
        #日期与时间段处理
        jd_consumer['action_time'] = jd_consumer['action_time'].astype(
            'datetime64')
        jd_consumer['action_hour'] = jd_consumer['action_time'].apply(
            lambda x: int(pd.datetime.strftime(x, '%H')))
        jd_consumer['action_date'] = jd_consumer['action_time'].apply(
            lambda x: pd.datetime.strftime(x, '%Y-%m-%d'))
        jd_consumer['action_date'] = jd_consumer['action_date'].astype(
            'datetime64')

        #将时段分为'凌晨'、'上午'、'中午'、'下午'、'晚上'
        jd_consumer['hour_bin'] = pd.cut(jd_consumer['action_hour'],
                                         bins=[-1, 5, 10, 13, 18, 24],
                                         labels=['凌晨', '上午', '中午', '下午', '晚上'])
        #初始化原始数据
        self.jd_consumer = jd_consumer
        #初始化标签列表
        users = jd_consumer['customer_id'].unique()
        self.user_labels = pd.DataFrame(users, columns=['customer_id'])
        #回收变量
        del jd_consumer
        gc.collect()

    def get_user_behavior_max(self):
        '''获取用户浏览、加购、购买最活跃的时间段/最多的商品'''
        for i in ['hour_bin', 'category']:
            for j in ['PageView', 'SavedCart', 'Order']:
                #对用户分组，统计各行为的次数
                df_gb = self.jd_consumer[
                    self.jd_consumer['type'] == j].groupby(
                        ['customer_id', i]).product_id.count().reset_index()
                df_gb.rename(columns={'product_id': 'counts'}, inplace=True)
                #统计每个用户各行为次数最多的时段
                df_gb_max = df_gb.groupby(
                    'customer_id').counts.max().reset_index()

                df_gb_max.rename(columns={'counts': 'counts_max'},
                                 inplace=True)
                df_gb = pd.merge(df_gb,
                                 df_gb_max,
                                 how='left',
                                 on='customer_id')
                #选取各用户行为次数最多的时段，如有并列最多的时段，用逗号连接
                df_gb_bin = df_gb.loc[
                    df_gb['counts'] == df_gb['counts_max'],
                    i].groupby(df_gb['customer_id']).aggregate(
                        lambda x: ','.join(x)).reset_index()
                #整合用户标签表中
                self.user_labels = pd.merge(self.user_labels,
                                            df_gb_bin,
                                            how='left',
                                            on='customer_id')
                self.user_labels.rename(
                    columns={i: str.lower(j) + '_' + i + '_max'}, inplace=True)

    def get_user_behavior_counts(self):
        '''获取用户近N天的购买、加购次数以及活跃天数'''
        for i in [7, 15]:
            for j in ['Order', 'SavedCart']:
                date_threshold = self.jd_consumer['action_date'].max(
                ) - timedelta(days=i)
                df_near_day = self.jd_consumer[
                    self.jd_consumer['action_date'] > date_threshold]
                df_counts = df_near_day[df_near_day['type'] == j].groupby(
                    'customer_id').product_id.count().reset_index(
                        name='count_' + str(i) + '_' + str.lower(j))
                self.user_labels = pd.merge(self.user_labels,
                                            df_counts,
                                            how='left',
                                            on='customer_id')

    def get_user_behavior_day(self):
        '''获取近N天行为活跃天数'''
        for i in [7, 15]:
            date_threshold = self.jd_consumer['action_date'].max() - timedelta(
                days=i)
            df_near_day = self.jd_consumer[
                self.jd_consumer['action_date'] > date_threshold]
            df_counts_active = df_near_day.groupby(
                'customer_id')['action_date'].nunique().reset_index(
                    name='count_' + str(i) + '_active')
            self.user_labels = pd.merge(self.user_labels,
                                        df_counts_active,
                                        how='left',
                                        on='customer_id')
        '''获取最后一次浏览、加购、购买行为距今(2018-05-01)天数'''
        for j in ['PageView', 'Savedcart', 'Order']:
            df_days = self.jd_consumer[self.jd_consumer['type'] == j].groupby(
                'customer_id')['action_date'].max().apply(
                    lambda x: (pd.datetime.strptime('2018-05-01', '%Y-%m-%d') -
                               x).days).reset_index(name=str.lower(j) + '_' +
                                                    'days')
            self.user_labels = pd.merge(self.user_labels,
                                        df_days,
                                        how='left',
                                        on='customer_id')
        '''获取最近两次购买时间间隔'''
        df_interval_order = self.jd_consumer[
            self.jd_consumer['type'] == 'Order'].groupby(
                ['customer_id',
                 'action_date']).product_id.count().reset_index()
        df_interval_order['action_date'] = df_interval_order[
            'action_date'].astype('datetime64')
        interval_order = df_interval_order.groupby(
            'customer_id')['action_date'].apply(lambda x: x.sort_values().diff(
                1).dropna().head(1)).reset_index()
        interval_order['action_date'] = interval_order['action_date'].apply(
            lambda x: x.days)
        interval_order.drop('level_1', axis=1, inplace=True)
        self.user_labels = pd.merge(self.user_labels,
                                    interval_order,
                                    how='left',
                                    on='customer_id')

    def get_user_behavior_labels(self):
        '''是否浏览/加购未购买'''
        for i in [['PageView', 'Order'], ['SavedCart', 'Order']]:
            view_order = self.jd_consumer[self.jd_consumer['type'].isin(i)]
            view_not_order = pd.pivot_table(
                view_order,
                index=['customer_id', 'product_id'],
                columns=['type'],
                values=['action_date'],
                aggfunc=['count'])
            view_not_order.columns = [i[0], i[1]]
            view_not_order.fillna(0, inplace=True)
            col = str.lower(i[0] + '_not_' + i[1])
            view_not_order[col] = 0
            view_not_order.loc[(view_not_order[i[0]] > 0) &
                               (view_not_order[i[1]] == 0), col] = 1
            view_not_order = view_not_order.groupby(
                'customer_id')[col].sum().reset_index()
            self.user_labels = pd.merge(self.user_labels,
                                        view_not_order,
                                        how='left',
                                        on='customer_id')
            self.user_labels[col] = self.user_labels[col].apply(
                lambda x: '是' if x > 0 else '否')
        '''是否复购/购买商品单一用户'''
        for j in ['order_again', 'order_single']:
            if j == 'order_again':
                df_order = self.jd_consumer[
                    self.jd_consumer['type'] == 'Order'].groupby(
                        'customer_id')['product_id'].count().reset_index(
                            name='order_again')
            else:
                df_order = self.jd_consumer[
                    self.jd_consumer['type'] == 'Order'].groupby(
                        'customer_id').category.nunique().reset_index(
                            name='order_single')
            self.user_labels = pd.merge(self.user_labels,
                                        df_order,
                                        how='left',
                                        on='customer_id')
            self.user_labels[j].fillna(-1, inplace=True)
            #未购买的用户标记为‘未购买’，有购买未复购/购买商品单一的用户标记为‘否’，有复购/购买商品不单一的用户标记为‘是’
            self.user_labels[j] = self.user_labels[j].apply(
                lambda x: '是' if x > 1 else '否' if x == 1 else '未购买')


if __name__ == '__main__':
    H = Handle()
    H.get_user_behavior_max()
    H.get_user_behavior_counts()
    H.get_user_behavior_day()
    H.get_user_behavior_labels()
    print(self.user_labels.head())
    H.user_labels.to_excel('../data/京东消费者用户画像标签2018-04.xlsx',index=False)
