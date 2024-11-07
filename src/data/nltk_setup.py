"""
Running this file brings up a window to download NLTK resources.
Shouldn't be necessary to run this to run data_collection.py, but if running that causes errors, try running this.
"""
def main():
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download()

if __name__ == '__main__':
    main()