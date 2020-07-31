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
        jd_consumer['action_time'] = jd_consumer['action_time'].astype(
            'datetime64')
        jd_consumer['action_date'] = jd_consumer['action_time'].apply(
            lambda x: pd.datetime.strftime(x, '%Y-%m-%d'))
        jd_consumer['action_date'] = jd_consumer['action_date'].astype(
            'datetime64')
        #初始化原始数据
        self.jd_consumer = jd_consumer[jd_consumer['gender'] != 'U']
        #self.jd_consumer = jd_consumer.loc[jd_consumer['gender']!='U',:]

    def get_tag_weight_tfidf(self, label):
        '''使用TF-IDF算法计算标签权重'''
        # 计算每个用户身上每个标签的个数
        user_tag_count_per = self.jd_consumer.groupby(
            [label,
             'category'])['action_time'].count().reset_index(name='weight_m_p')

        # 计算每个用户身上的标签总数
        user_tag_count_total = self.jd_consumer.groupby(
            [label])['category'].count().reset_index(name='weight_m_s')

        tag_weight_tfidf = pd.merge(user_tag_count_per,
                                    user_tag_count_total,
                                    how='left',
                                    on=label)

        # 每个标签的行为数
        type_tag_count_per = tag_weight_tfidf.groupby(
            ['category']).weight_m_p.sum().reset_index(name='weight_w_p')

        # 所有标签的总和
        tag_weight_tfidf['weight_w_s'] = user_tag_count_per['weight_m_p'].sum()

        tag_weight_tfidf = pd.merge(tag_weight_tfidf,
                                    type_tag_count_per,
                                    how='left',
                                    on='category')
        if label == 'customer_id':
            # 应用TF-IDF计算标签权重
            tag_weight_tfidf['tfidf_ratio'] = (
                tag_weight_tfidf['weight_m_p'] / tag_weight_tfidf['weight_m_s']
            ) * (np.log10(tag_weight_tfidf['weight_w_s'] /
                          tag_weight_tfidf['weight_w_p']))
            behavior_count = self.jd_consumer.groupby([
                label, 'type', 'category'
            ])['action_time'].count().reset_index(name='behavior_count')

            tag_weight_tfidf = pd.merge(self.jd_consumer,
                                        tag_weight_tfidf,
                                        how='left',
                                        on=[label,
                                            'category']).reset_index(drop=True)
            tag_weight_tfidf = pd.merge(tag_weight_tfidf,
                                        behavior_count,
                                        how='left',
                                        on=[label, 'type',
                                            'category']).reset_index(drop=True)
        else:
            # 使用TF-IDF算法计算每个性别对每个标签的偏好权重值
            tag_weight_tfidf['tfidf_ratio'] = (
                tag_weight_tfidf['weight_m_p'] / tag_weight_tfidf['weight_m_s']
            ) * (tag_weight_tfidf['weight_w_s'] /
                 tag_weight_tfidf['weight_w_p'])
        return tag_weight_tfidf

    def get_behavior_type_weight(self):
        '''建立行为类型权重维表'''
        tag_weight_tfidf = self.get_tag_weight_tfidf('customer_id')
        #浏览行为，权重0.3，收藏行为，权重0.5，加购行为，权重1，购买行为，权重1.5，评论行为，权重2
        tag_weight_tfidf['act_weight_init'] = 0.3
        dic = {'PageView': 0.3, 'SavedCart': 1, 'Order': 1.5, 'comment': 2}
        for k, v in dic.items():
            tag_weight_tfidf.loc[tag_weight_tfidf['type'] == k,
                                 'act_weight_init'] = v
        '''电商项目中，加购行为的权重不随着时间的增长而衰减，而购买、浏览、收藏、评论随着时间的推移，
        其对当前的参考性越来越弱，因此权重会随着时间的推移越来越低'''
        ## 计算用户标签权重
        tag_weight_tfidf['time_reduce_ratio'] = 1
        tag_weight_tfidf.loc[
            tag_weight_tfidf['type'] != 'SavedCart',
            'time_reduce_ratio'] = tag_weight_tfidf.loc[
                tag_weight_tfidf['time_reduce_ratio'] != 'SavedCart',
                'action_date'].apply(lambda x: self.weight_time_reduce(x))
        # 标签总权重 = 行为类型权重*衰减系数*行为数*TFIDF标签权重
        tag_weight_tfidf['act_weight'] = tag_weight_tfidf[
            'act_weight_init'] * tag_weight_tfidf[
                'time_reduce_ratio'] * tag_weight_tfidf[
                    'behavior_count'] * tag_weight_tfidf['tfidf_ratio']
        return tag_weight_tfidf

    def weight_time_reduce(self, action_date):
        #标签权重衰减函数
        date_interval = pd.datetime.strptime('2018-05-01',
                                             '%Y-%m-%d') - action_date
        date_interval = date_interval.days
        time_reduce_ratio = np.exp(date_interval * (-0.1556))
        return time_reduce_ratio

    def get_user_recommend_tag(self):
        # 要计算两两标签的相似性
        user_type = self.jd_consumer[['customer_id', 'category']]
        ## 计算两两标签共同对应的用户数
        # 将两表正交，得到每个用户下，其所有标签的的两两组合
        user_type_mix = pd.merge(user_type, user_type, on='customer_id')

        # 删除重复值，即同一用户由上述正交得到的数据表中，两个标签为同一标签的数据
        user_type_drop = user_type_mix.drop(labels=user_type_mix[
            user_type_mix['category_x'] == user_type_mix['category_y']].index,
                                            axis=0)

        # 用两个标签分组，计算用户数，即每两个标签同时出现在不同的用户中的个数
        user_tag = user_type_drop.groupby([
            'category_x', 'category_y'
        ])['customer_id'].count().reset_index(name='counts_common')

        ## 计算每个标签对应的用户数

        # 计算每一个标签对应的不同的用户数，即每个标签出现在不同的用户中的个数
        type_user_count = user_type.groupby([
            'category'
        ])['customer_id'].nunique().reset_index(name='counts_category_user')

        # 计算标签1有关的用户数
        user_tag = pd.merge(user_tag,
                            type_user_count,
                            how='left',
                            left_on='category_x',
                            right_on='category').drop('category_x', axis=1)

        user_tag.rename(columns={
            'counts_category_user': 'counts_category_x',
            'category': 'category_x'
        },
                        inplace=True)

        # 计算标签2有关的用户数
        user_tag = pd.merge(user_tag,
                            type_user_count,
                            how='left',
                            left_on='category_y',
                            right_on='category').drop('category_y', axis=1)

        user_tag.rename(columns={
            'counts_category_user': 'counts_category_y',
            'category': 'category_y'
        },
                        inplace=True)

        ## 计算两两标签之间的相似性

        # 余弦相似度计算两两标签的相关性
        user_tag['power'] = user_tag['counts_common'] / np.sqrt(
            user_tag['counts_category_x'] * user_tag['counts_category_y'])

        ## 对每个用户的历史标签权重加总

        # 对用户、标签进行分组，计算每个用户每个标签的权重和
        tag_weight_tfidf = self.get_behavior_type_weight()
        user_tag_weight_sum = tag_weight_tfidf.groupby(
            ['customer_id', 'category'])['act_weight'].sum().reset_index()

        ## 计算推荐给用户的相关标签

        # 将用户与所有与其有关的标签作对应
        user_peasona_tag = pd.merge(user_tag_weight_sum,
                                    user_tag,
                                    how='left',
                                    left_on='category',
                                    right_on='category_x').drop('category',
                                                                axis=1)
        # 计算推荐得分值  得分值 = 行为权重*相关性
        user_peasona_tag['recommend'] = user_peasona_tag[
            'act_weight'] * user_peasona_tag['power']

        # 对所有数据按得分值排序，再按’user_id'分组，得到每个用户有关的得分值最高的10个标签
        user_peasona_tag_total = user_peasona_tag.sort_values(
            'recommend', ascending=False).groupby(['customer_id']).head(10)
        #user_peasona_tag_total.drop(['action_hour','action_time','action_date','type','gender']
        #',axis=1,inplace=True)
        user_peasona_tag_total.to_excel(
            '../data/京东消费者个性化标签偏好.xlsx', index=False)
        print(user_peasona_tag_total.head())

    def get_group_recommend_tag(self):
        group_tag_weight_tfidf = self.get_tag_weight_tfidf('gender')
        group_weight_tag = group_tag_weight_tfidf.sort_values(
            'tfidf_ratio', ascending=False).groupby(['gender']).head(10)
        group_weight_tag.to_excel('../data/京东消费者群体标签偏好.xlsx',
                                  index=False)
        print(group_weight_tag.info())


if __name__ == '__main__':
    H = Handle()
    H.get_user_recommend_tag()
    H.get_group_recommend_tag()
