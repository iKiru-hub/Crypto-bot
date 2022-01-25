# Crypto-bot
A Binance-API interface in which insert a trading strategy of your choice for buy/sell coins

It is provided an Environment class that takes care of defining the interface, namely getting the market data and executing the buy/sell orders. 
The Agent class instead is responsible for making the decision of buy/sell coins (based on a strategy that the user must define, the default is random!) based on HLCV data (high-low-close-volume) of a certain coin at a certain time.

# Run Mode
Two run-mode are available: 
  - FAKE (default) : the agent is applied to historic data (past 2 days), the results is the average performance/profit over 100 sessions
  - REAL : the agent is applied to real-time data and the orders executed use the available USDT availability in the user's wallet

# Candlesticks
The default candlesticks are: 5m for 2 days (customizeable) 

# TODO
In order for the program to connect to Binance, and presuming you already possess an account, you have to:
  - save the two scripts <plain_bot.py> and <run.py> in a given folder
  - obtain the api key and secret key for accessing your Binance API client
  - save the two password in a file named <nonfile.txt> in the same folder and separated by an arbitrary string #k of your choice, e.g. myapikey$mysecretkey where the arbitrary string is #k
  - run the program main.py with defined arguments #k (the string you define earlier) #env (OS environment: 'win' for Windows or 'linux') #ulogs (the period for updating the logs of the session, in minutes) #mode (fake: 1, or real: 0)

# reminder
The crypto-market is a very odd place, good luck with whatever algorithm you go with. 
The bot will run eternally until the program breaks somehow (bugs?), the connections is lost, Binance cut the connection or no more USDT are available.

Be cautious, you shouldn't risk more money you are willing to lose!
