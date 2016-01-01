{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RebindableSyntax #-}

module Course.FileIO where

import Course.Core
import Course.Applicative
import Course.Monad
import Course.Functor
import Course.List

{-

Useful Functions --

  getArgs :: IO (List Chars)
  putStrLn :: Chars -> IO ()
  readFile :: Chars -> IO Chars
  lines :: Chars -> List Chars
  void :: IO a -> IO ()

Abstractions --
  Applicative, Monad:

    <$>, <*>, >>=, =<<, pure

Problem --
  Given a single argument of a file name, read that file,
  each line of that file contains the name of another file,
  read the referenced file and print out its name and contents.

Example --
Given file files.txt, containing:
  a.txt
  b.txt
  c.txt

And a.txt, containing:
  the contents of a

And b.txt, containing:
  the contents of b

And c.txt, containing:
  the contents of c

$ runhaskell FileIO.hs "files.txt"
============ a.txt
the contents of a

============ b.txt
the contents of b

============ c.txt
the contents of c

-}

-- /Tip:/ use @getArgs@ and @run@
main ::
  IO ()
main = run =<< headOr "" <$> getArgs

type FilePath =
  Chars

-- /Tip:/ Use @getFiles@ and @printFiles@.
run ::
  Chars
  -> IO ()
run fpath = printFiles =<< getFiles (pure fpath)

getFiles ::
  List FilePath
  -> IO (List (FilePath, Chars))
getFiles fpaths = sequence $ getFile <$> fpaths

getFile ::
  FilePath
  -> IO (FilePath, Chars)
getFile fpath = (\x -> (fpath, x)) <$> readFile fpath

printFiles ::
  List (FilePath, Chars)
  -> IO ()
printFiles fs = flattenIOs $ uncurry printFile <$> fs

printFile ::
  FilePath
  -> Chars
  -> IO ()
printFile p c = flattenIOs $ putStrLn <$> (p :. lines c)

flattenIOs :: List (IO ()) -> IO ()
flattenIOs ios = void $ sequence ios

