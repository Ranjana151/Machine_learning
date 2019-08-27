from textblob import TextBlob

feedback1="The food at Radison was not good at taste"
feedback2="The food at Radison was very great"
blob1=TextBlob(feedback1)
blob2=TextBlob(feedback2)
print(blob1.sentiment)
print(blob2.sentiment)