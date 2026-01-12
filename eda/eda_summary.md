# EDA Summary

My data set is team and game statistics from every game in modern NBA history. I choose this realm of data because I'm extremely interested in basketball, and have the requisite domain knowledge to possibly work well with the information. Aside from personal interest, it is also just an extremely clean data set, with complete records and uniform structure. There are no missing or null values unless they are N/A, usually due to being from an era before that field was recored. This is not hard to believe for pro sports data, which is very well maintained and public. It's also because my sources curated the information very well. My data set comes from three seperate sources, one is a library connected to the nba API called *nba_api*, and the other two were Kaggle sets formed by Eoin A Moore and Wyatt Walsh, respectively. Team-only statistics were derived from these sets, and after combining key data fields and information across the three sources into a single database, it formed a database which covered every NBA game spanning 41 seasons, from 1983 - 2024.

There were a total of 49,626 games present in the data set, and there were team box-score statistics including points (PTS), offensive rebounds (OREB), defensive rebounds (DREB), total assists (AST), total steals (STL), total blocks (BLK), etc. In order to attempt to analyze and compare the effectiveness of teams, as is my intention for a model, I computed and imputed advanced stats, ranging from true shooting percentage, assist-to-turnover ratio, team pace, possession count, net rating, home game advantage, outcome of previous game, and back-to-back status.

<img height="350" alt="home away win rate" src="https://github.com/user-attachments/assets/68ebcf42-2792-49a2-96ba-e347a0d08f15" />

<img height="350" alt="b2b win rate" src="https://github.com/user-attachments/assets/46ae7336-2851-4299-9250-efab8f6473dd" />

Upon intial exploration, I charted the win rate of home and away teams as well as for teams playing on a back-to-back, which is two or more games on consecutive nights. Interestingly, the data clearly supported intuition, which is that home teams win discernably more often than the away team. The home team won about 60% of the time, which is a significant statistical advantage. Conversly, the teams playing back-to-backs lost about 56% of the time, indicating that one team in the matchup playing consecutive games very likely has physical and possibly mental disadvantages.

I thought that these statistics, along with the normal box scores and advanced stats, would be good candidates for a predictive model of team success across a season. However, it may be tricky because if the model is aware of both teams statistics it could simply deduce the winner by the point differential. The winner will be plainly embedded in between the statistics of the two teams. So perhaps a model which only looks at a single teams past performance for predictive purposes.

<p align="center">
<img height="200" alt="pts avg over time" src="https://github.com/user-attachments/assets/667876ab-d8b6-499b-a17a-0b781bfdbb2e" />
<img height="200" alt="TS avg over time save for later" src="https://github.com/user-attachments/assets/a83adf39-4159-45f0-b148-680df35c84c9" />
</p>
<p align="center">
<img height="200" alt="rebounding avg over time" src="https://github.com/user-attachments/assets/a6d3554e-2e19-47d6-9184-ca49abcc9aa3" />
</p>

Looking at historical trends and plotting them for a few fundamental statistics we see that points, true shooting percentage, and rebounding all follow a very similar trend and are tightly coupled. The stats begin very high in the mid 1980s but teeter down to trough by the late 1990s, before climbing back up to all time highs. It is rather unexpected that the values of these stats today are comparable to their 1980s values. Perhaps these stats are all tightly coupled because there is an underlying causal statistic between them, such as pace of game. A challenge regarding this phenomenon is that if I were to train on the full span of the data, the game and style of play changes so much between eras that training which involves the 1990s may have poor performance in predicting the 2020s. A possible solution may be to be era independent, or decrease the training set size and temporal coverage to more modern times. Presently, this seems like one of the larger data decisions regarding the actual training and effectiveness of a potential model.
