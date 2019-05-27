data = read.csv('D:\\GitHub\\Airbnb\\Recommendation\\Tasmania-FinalDataset.csv')

library(cluster)
library(dplyr)
library(ggplot2)
library(readr)
library(Rtsne)

gower_dist <- daisy(data[,3:32], metric = "gower")
gower_mat <- as.matrix(gower_dist)
listing <- data[,1:2]

max(gower_mat)

#Create an empty data frame to store recommendations
recommendations = data.frame(listing_id = numeric,recommendation = numeric, g_dist = double)

#Look at similar listings based on gower distance
for(i in 1:nrow(listing))
{
  #Retrieve sorted array of gower distances for the current listing
  sorted = gower_mat[i,] %>% sort(, decreasing = FALSE, index.return = TRUE)
  #Retrieve indexes for the next 10 closest listings
  idx = sorted$ix[2:11]
  #Use indexes to retrieve corresponding listing ids
  recom = listing[idx,'listing_id']
  recommendations = rbind(recommendations, cbind(data$listing_id[1],recom,sorted$x[2:11]))
}

write.csv(recommendations, 'D:\\GitHub\\Airbnb\\Recommendation\\Nearest Neighbors.csv')

reviewer_listing = read.csv('D:\\GitHub\\Airbnb\\Recommendation\\Reviewer-Listing.csv')

reviewers = unique(reviewer_listing[,1])

#Create an empty data frame to store avg gower distance between listings of single user
reviewer_avg_dist = data.frame(reviewer_id = numeric,avg_g_dist = double)

#Retrieve distances of listings booked by same user
for(i in 1:length(reviewers))
{
 #Extract all listings visiting by single user
 r_listings = reviewer_listing[reviewer_listing$reviewer_id==reviewers[i],'listing_id']
 #Generate all possible combinations of listings
 if(length(r_listings) > 2)
 {
   comb = combn(r_listings,2)
   #Loop through the combinations and generate avg gower distance
   gdist = vector()
   for(j in 1:ncol(comb))
   {
     #Gower distance of current combination of listings
     gdist = c(gdist, gower_mat[which(data$listing_id==comb[1,j]),which(data$listing_id==comb[2,j])])
   }
   reviewer_avg_dist = rbind(reviewer_avg_dist, cbind(reviewers[i], mean(gdist)))
 }
}

nrow(reviewer_avg_dist)
hist(reviewer_avg_dist$V2)

write.csv(reviewer_avg_dist, 'D:\\GitHub\\Airbnb\\Recommendation\\Reviewers_Avg_Dist.csv')

------------------------------------------------------------------------------------------
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

#Look at cluster values
pam_fit$clustering[1337]

