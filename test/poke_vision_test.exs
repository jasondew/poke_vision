defmodule PokeVisionTest do
  use ExUnit.Case
  doctest PokeVision

  test "greets the world" do
    assert PokeVision.hello() == :world
  end
end
