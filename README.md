

### Implementation and execution ###

A reasonably complex dataset with some interesting problems to tackle, especially in data wrangling and data conversion. Column encoding is also required for doing correlation analysis, though using onehot encoding is a stretch for  columns that qualify, viz., Marital_Status and Country. The education column is suitable candidate for ordinal encoding. I have only used ordinal and label encoding for this project.

I have added about 10 columns to the existing 28 to facilitate analysis and visualizations. Esentially all encoded columns are added as new columns and
certain extra columns were created to hod age(calculated from birth year) and to hold ranges such for age and income (AgeBracket and IncomeBracket). This gives a better visualization. The following extra columns were added

The initial data shape is  (2240, 28)

Added the following columns for analysis

    1.TotalAcrossProducts (Sum of all spends)
	2.TotalAcrossChannels (sum of spends across channels)
	3. OffStorePurchases ( Purchases other than store purchases)
	4. ECountry (Encoded country column)
	5. EEducation (Encoded education column)
	6. EMarital_Status (Encoded marital status column)
	7.Age (Age , calculated from birth year )
	8. AgeBracket (3 age brackts viz., < 30 , 30 - 60 and > 60 )
	9. EAgeBracket( Encoded AgeBracket )
	10. IncomeBracket( 4 income brackets viz., < 25K , 25-45K, 45-65K and > 65K )
	11. EIncomeBracket (Encoded IncomeBracket)
	12.TotalChildren ( Total no of children, Kidhome + Teenhome )

Dropped three rows for age outliers. The data has three ages [124, 125, 131].
So the final data shape is  (2237, 40)

#### Suggested execution environment : google colab. ####