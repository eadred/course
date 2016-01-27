{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Course.Compose where

import Course.Core
import Course.Functor
import Course.Applicative
import Course.Monad

-- Exactly one of these exercises will not be possible to achieve. Determine which.

newtype Compose f g a =
  Compose (f (g a))

-- Implement a Functor instance for Compose
instance (Functor f, Functor g) =>
    Functor (Compose f g) where
  f <$> Compose x = Compose $ (f <$>) <$> x

instance (Applicative f, Applicative g) =>
  Applicative (Compose f g) where
-- Implement the pure function for an Applicative instance for Compose
  pure x = Compose $ pure (pure x)
-- Implement the (<*>) function for an Applicative instance for Compose
  Compose f <*> Compose x = let f' = (<*>) <$> f in -- f' :: f (g a -> g b) -> g a -> g b
                                 Compose $ f' <*> x

instance (Monad f, Monad g) =>
  Monad (Compose f g) where
-- Implement the (=<<) function for a Monad instance for Compose
  (=<<) =
    error "todo: Course.Compose (<<=)#instance (Compose f g)"
