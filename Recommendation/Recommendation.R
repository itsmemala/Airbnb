data = read.csv('D:\\PGP-BABI\\Capstone\\Airbnb\\Tasmania-FinalDataset.csv')

library(cluster)
library(dplyr)
library(ggplot2)
library(readr)
library(Rtsne)

gower_dist <- daisy(data[,3:32], metric = "gower")
gower_mat <- as.matrix(gower_dist)
listing <- data[,1:2]

#Look at similar listings based on gower distance
listing[data$listing_id==12092875,'X']
sorted = gower_mat[1337,] %>% sort(, decreasing = TRUE)
sorted[1] #how to extract the name (i.e row id 2629)
listing[data$X==2629,'listing_id']

#plot silhoutte width to find optimum number of clusters
sil_width <- c(NA)
for(i in 5:15){  
  pam_fit <- pam(gower_dist, diss = TRUE, k = i)  
  sil_width[i] <- pam_fit$silinfo$avg.width  
}
plot(1:15, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:15, sil_width)

k <- 9
pam_fit <- pam(gower_dist, diss = TRUE, k)
pam_results <- data %>%
  mutate(cluster = pam_fit$clustering) %>%
  group_by(cluster) %>%
  do(the_summary = summary(.))

dd <- cbind(data, cluster = pam.res$cluster)
head(dd, n = 3)

#Look at cluster values
pam_fit$clustering[1337]

