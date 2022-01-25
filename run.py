import os
import sys
import argparse
import time
import plain_bot as pb



if __name__ == '__main__':

    ################################################################################################
    ### PARSER

    parser = argparse.ArgumentParser(description='initialization of the trader')

    parser.add_argument('k', type=str, help='value for k')
    parser.add_argument('env', nargs='?', const=1, type=str, default='pi',
                        help='environment in which the program will operate: default <pi> else <win>')
    parser.add_argument('ulogs', nargs='?', const=1, type=int, default=30,
                        help='time update for the logs')
    parser.add_argument('fake', nargs='?', const=1, type=int, default=1,
                        help='run with fake money and 5m/2days (type 1, default) or real money (type 0)')
    args = parser.parse_args()


    # check OS environment spelling
    if args.env == 'win' or args.env == 'linux':
        pass
    else:
        raise TypeError(f'environment <{args.env}> un-available, only <linux> and <win> are allowed')

    ################################################################################################
    ### CHECK DIRECTORY

    path = os.getcwd()
    files = os.listdir(path)

    # credential file
    if not files.__contains__('nonfile.txt'):
        print('\n it looks like your Binance credentials are unreachable, <nonfile.txt> not found!')
        sys.exit()

    # session folder
    if not files.__contains__('sessions'):
        if args.env == 'win':
            os.mkdir(path + '\\sessions')
        else:
            os.mkdir(path + '//sessions')

        print('\n <sessions> folder built')


    ################################################################################################
    ### LOAD & ACCESS

    file = 'nonfile.txt'

    # access to Binance
    with open(file, 'r') as f:
        jello = f.read()

    api_key, api_secret = jello.split(args.k)

    if args.env == 'win':
        os.system('cls')
    else:
        os.system('clear')

    print('\n<access successful>')
    time.sleep(0.5)


    ##################################################################################################
    ### RUN

    # environment set up | change candlestick timestamp (dt) or period (t0) as you prefer
    main = pb.Env(capital=100,
                  a_k=api_key, a_s=api_secret,
                  env=args.env,
                  dt='5m',
                  t0='2 days',
                  ulogs=args.ulogs,
                  fake=args.fake == 1)

    time.sleep(1.)

    # run
    if args.fake == 1:
        main.avg_fake(N=100)  #
    else:
        main.main()
