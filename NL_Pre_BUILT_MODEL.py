from google.cloud import language_v1
#google-cloud-language
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
    r'C:\Users\HEMANTH KUMAR K\Desktop\SERVICE ACCOUNT KEYS\machine-learning-322713-babbff790858.json'

def sample_classify_text(text_content):
    """
    Classifying Content in a String

    Args:
      text_content The text content to analyze. Must include at least 20 words.
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    response = client.classify_text(request = {'document': document})
    # Loop through classified categories returned from the API
    for category in response.categories:
        # Get the name of the category representing the document.
        # See the predefined taxonomy of categories:
        # https://cloud.google.com/natural-language/docs/categories
        print(u"Category name: {}".format(category.name))
        # Get the confidence. Number representing how certain the classifier
        # is that this category represents the provided text.
        print(u"Confidence: {}".format(category.confidence))

content = "This maid of mine who comes to wash dishes at home is usually late. " \
          "This morning I was in a hurry to go to the bank for some important work. " \
          "Well what do you know! The maid came right on time. The dishes were cleaned. " \
          "I did not want to come home to a mess."

sample_classify_text(content)