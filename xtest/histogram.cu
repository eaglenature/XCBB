#include "common/xtestrunner.h"

#include <xcbb/xcbb.h>

class HistogramTest: public CudaTest
{
protected:
    HistogramTest() {}
    virtual ~HistogramTest() {}
    virtual void SetUp()
    {
        CudaTest::SetUp();
        std::srand(time(0));
    }
    virtual void TearDown() {}

protected:

    void CreateSample0(std::vector<uint>& keys)
    {
        for (uint i = 0; i < keys.size(); ++i) keys[i] = rand() % 1024;
    }

    void CreateSample1(std::vector<uint>& keys)
    {
        ASSERT_GT(keys.size(), 100);
        for (uint i = 0;  i < 10;          ++i) keys[i] = 1000;
        for (uint i = 10; i < 30;          ++i) keys[i] = 1001;
        for (uint i = 30; i < 60;          ++i) keys[i] = 1002;
        for (uint i = 60; i < keys.size(); ++i) keys[i] = rand() % 1000;
    }

    void CreateSample2(std::vector<uint>& keys)
    {
        const uint constkey = rand() % 1024;
        for (uint i = 0; i < keys.size(); ++i) keys[i] = constkey;
    }

    void CreateSample3(std::vector<uint>& keys)
    {
        const uint constkey1 = rand() % 1024;
        const uint constkey2 = rand() % 1023 + 1;
        const uint range = rand() % 2000 + 1;
        const uint n = keys.size();
        ASSERT_TRUE(range < n);
        for (uint i = 0; i < range; ++i) keys[i] = constkey1;
        for (uint i = range; i < n; ++i) keys[i] = constkey2;
    }

    void CreateSample4(std::vector<uint>& keys)
    {
        for (uint i = 0; i < keys.size(); ++i) keys[i] = 0;
    }

    void SerialHistogram(const std::vector<uint>& keys, std::vector<uint>& histogram)
    {
        for (uint k = 0; k < keys.size(); ++k)
        {
            uint key = keys[k];
            ASSERT_LT(key, histogram.size());
            ASSERT_GE(key, 0);
            ++histogram[key];
        }
    }
};


TEST_F(HistogramTest, Test0)
{
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
